
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


# kernel path: /tmp/torchinductor_youkaichao/o5/co5muf266xxev5mnc6naxptfz3cggzv3olkzd7t4xzts45jac6aw.py
# Source Nodes: [hidden_states_12, hidden_states_4, query_states], Original ATen: [aten.bernoulli]
# hidden_states_12 => bernoulli_3
# hidden_states_4 => bernoulli_2
# query_states => bernoulli
triton_poi_fused_bernoulli_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bernoulli_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
    tl.store(out_ptr2 + (x0), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg3hv6btynihi7wss4wspjfionvyzvfelmv7llupsqbxs27jlq7.py
# Source Nodes: [add, embeddings, embeddings_1, embeddings_3, hidden_states_1, mean, mul, position_embeddings, pow_1, query_states, sqrt, sub, variance], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub]
# add => add_1
# embeddings => embedding
# embeddings_1 => add
# embeddings_3 => add_2
# hidden_states_1 => div
# mean => mean
# mul => mul
# position_embeddings => embedding_1
# pow_1 => pow_1
# query_states => convert_element_type, full_default_1, mul_2, sub_2, where
# sqrt => sqrt
# sub => sub
# variance => mean_1
triton_red_fused__to_copy_add_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp17 = tl.load(in_ptr1 + (r1 + (768*tmp16)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp5 + 512
        tmp19 = tmp5 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp5)
        tl.device_assert(((0 <= tmp20) & (tmp20 < 512)) | ~xmask, "index out of bounds: 0 <= tmp20 < 512")
        tmp21 = tl.load(in_ptr3 + (r1 + (768*tmp20)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp17 + tmp21
        tmp23 = 768.0
        tmp24 = tmp12 / tmp23
        tmp25 = tmp22 - tmp24
        tmp26 = tmp25 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tl.store(out_ptr1 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-07
    tmp33 = tmp31 + tmp32
    tmp34 = tl.sqrt(tmp33)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp34, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp35 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(out_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = 1.0
        tmp37 = tmp36 - tmp35
        tmp38 = (tmp37 != 0)
        tmp41 = tmp40 / tmp34
        tmp42 = tmp39 * tmp41
        tmp44 = tmp42 + tmp43
        tmp45 = 0.0
        tmp46 = tl.where(tmp38, tmp45, tmp44)
        tmp47 = 1.1111111111111112
        tmp48 = tmp46 * tmp47
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp38, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (768*x0)), tmp48, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4clci7clzioy5yjxnw77y6pkfywhy764sdjcrpg7oc37ikg5onl.py
# Source Nodes: [query_layer_1, query_layer_2, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
# query_layer_1 => add_3
# query_layer_2 => div_1
# scale => full_default_2
triton_poi_fused_add_div_sqrt_transpose_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_sqrt_transpose_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (x0 + (64*x2) + (768*x1)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czkgajidm2az3rm7kez4soyylx6dl6f2e4ejgqk7nubpzcfnovqx.py
# Source Nodes: [attention_probs_1, attention_probs_3], Original ATen: [aten.bernoulli]
# attention_probs_1 => bernoulli_1
# attention_probs_3 => bernoulli_4
triton_poi_fused_bernoulli_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bernoulli_3', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvuieodaligpjeanorpwuozisaw2ahmy6iahg5s6oezh44pdu5wj.py
# Source Nodes: [attention_probs, attention_probs_1, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
# attention_probs => amax, div_2, exp, full_default_3, full_default_4, sub_3, sum_1, where_1, where_2
# attention_probs_1 => convert_element_type_2, mul_5, sub_4, where_3
# query_states => full_default_1
triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp14 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
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
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = (tmp16 != 0)
    tmp18 = tmp9 / tmp13
    tmp19 = 0.0
    tmp20 = tl.where(tmp1, tmp19, tmp18)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp17, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp23, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu64u42jfqpwtchgbzy3bk3qh6gcvdq6smccgfy2ed4wiqwj7ptc.py
# Source Nodes: [value_layer_1], Original ATen: [aten.add, aten.transpose]
# value_layer_1 => add_4
triton_poi_fused_add_transpose_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_transpose_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (x0 + (64*x2) + (768*x1)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tjqgnerg2jttdkx3o3t2wp5lc7e343tqrf5opyg5pcpaaeo73r.py
# Source Nodes: [hidden_states_3], Original ATen: [aten.view]
# hidden_states_3 => view_12
triton_poi_fused_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_6', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (32768*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hw/chwasttccqzel4tdofs5yjnjmidfrkmkbkks6w5qzdb4umg632g4.py
# Source Nodes: [add_4, add_5, attention_output, hidden_states_4, hidden_states_6, hidden_states_9, mean_3, mul_4, pow_2, query_states, sqrt_2, sub_2, variance_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
# add_4 => add_5
# add_5 => add_6
# attention_output => add_7
# hidden_states_4 => convert_element_type_3, mul_6, sub_5, where_4
# hidden_states_6 => div_3
# hidden_states_9 => view_14
# mean_3 => mean_2
# mul_4 => mul_7
# pow_2 => pow_2
# query_states => full_default_1
# sqrt_2 => sqrt_2
# sub_2 => sub_6
# variance_1 => mean_3
triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp4 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = (tmp2 != 0)
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp3, tmp7, tmp6)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 768.0
    tmp18 = tmp16 / tmp17
    tmp19 = tmp12 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp24 / tmp17
    tmp26 = 1e-07
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp30 = tmp19 / tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp3, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp28, xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqsyju2fv47s6bujukp6iflfi7dnyx2nhmfxsow3p3yteh4d6wl.py
# Source Nodes: [hidden_states_11, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_11 => view_16
# intermediate_output => add_8, erf, mul_10, mul_8, mul_9
triton_poi_fused_gelu_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/da/cdaoa3nwtrim3nhmwuqrk2jscagfdsmyg352bhmknboypjtdswjh.py
# Source Nodes: [add_7, add_8, attention_output, hidden_states_12, hidden_states_14, hidden_states_6, mean_6, mul_4, mul_5, pow_3, qp_1, query_states, query_states_1, sqrt_3, sub_4, variance_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
# add_7 => add_9
# add_8 => add_10
# attention_output => add_7
# hidden_states_12 => convert_element_type_4, mul_11, sub_8, where_5
# hidden_states_14 => div_4
# hidden_states_6 => div_3
# mean_6 => mean_4
# mul_4 => mul_7
# mul_5 => mul_12
# pow_3 => pow_3
# qp_1 => view_18
# query_states => full_default_1
# query_states_1 => add_11
# sqrt_3 => sqrt_3
# sub_4 => sub_9
# variance_2 => mean_5
triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp4 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = (tmp2 != 0)
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp3, tmp7, tmp6)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = 768.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp18 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp30 / tmp23
    tmp32 = 1e-07
    tmp33 = tmp31 + tmp32
    tmp34 = tl.sqrt(tmp33)
    tmp36 = tmp25 / tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp3, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp34, xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv46wo2rb54aafh3hgtiujxug225bbmoyuddyj3iiofp4qtkewe.py
# Source Nodes: [hidden_states_19, hidden_states_27, hidden_states_34, hidden_states_42], Original ATen: [aten.bernoulli]
# hidden_states_19 => bernoulli_5
# hidden_states_27 => bernoulli_6
# hidden_states_34 => bernoulli_8
# hidden_states_42 => bernoulli_9
triton_poi_fused_bernoulli_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bernoulli_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
    tl.store(out_ptr2 + (x0), tmp0, None)
    tl.store(out_ptr3 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3tprp3vqc3v67ynmnmi5n7th6tspectuffubxsvl4e2rwiqfxj.py
# Source Nodes: [hidden_states_169, hidden_states_177], Original ATen: [aten.bernoulli]
# hidden_states_169 => bernoulli_35
# hidden_states_177 => bernoulli_36
triton_poi_fused_bernoulli_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bernoulli_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2mvkabmcfp24pvys54xmpe3mfwv6m2q76mddj2fwmsktrdcxik.py
# Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
# start_logits_1 => clone_12
# start_loss => amax_12, exp_12, log, sub_100, sub_99, sum_13
triton_per_fused__log_softmax_clone_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldaxey4k63fzjbykrt54ixy5xzczm7wi2v53jitekilnntqdbor.py
# Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
# end_logits_1 => clone_13
# end_loss => amax_13, exp_13, log_1, sub_101, sub_102, sum_16
triton_per_fused__log_softmax_clone_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5bf4piun2vmrcdemfeo2aehdzartvxndkbtsow3oxghfhzmabp.py
# Source Nodes: [add_98, end_loss, end_positions, loss, query_states, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
# add_98 => add_111
# end_loss => convert_element_type_50, div_50, ne_3, neg_1, sum_17, sum_18, where_64
# end_positions => clamp_max_1, clamp_min_1
# loss => div_51
# query_states => full_default_1
# start_loss => convert_element_type_49, div_49, full_default_86, ne, neg, sum_14, sum_15, where_62
# start_positions => clamp_max, clamp_min
triton_poi_fused_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: '*fp32', 7: '*i1', 8: '*i64', 9: '*i1', 10: '*i64', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp9 = triton_helpers.maximum(tmp8, tmp2)
    tmp10 = triton_helpers.minimum(tmp9, tmp4)
    tmp11 = tmp10 != tmp4
    tmp12 = tl.where(tmp6, tmp5, tmp2)
    tmp13 = tmp12 + 512
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 512), "index out of bounds: 0 <= tmp15 < 512")
    tmp16 = tl.load(in_ptr2 + (tmp15), None, eviction_policy='evict_last')
    tmp17 = -tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tmp6.to(tl.int64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tl.where(tmp11, tmp10, tmp2)
    tmp24 = tmp23 + 512
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert((0 <= tmp26) & (tmp26 < 512), "index out of bounds: 0 <= tmp26 < 512")
    tmp27 = tl.load(in_ptr3 + (tmp26), None, eviction_policy='evict_last')
    tmp28 = -tmp27
    tmp29 = tl.where(tmp11, tmp28, tmp18)
    tmp30 = tmp11.to(tl.int64)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp22 + tmp32
    tmp34 = 2.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp23, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166 = args
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
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
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
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (50265, 768), (768, 1))
    assert_size_stride(primals_76, (512, 768), (768, 1))
    assert_size_stride(primals_77, (2304, 768), (768, 1))
    assert_size_stride(primals_78, (768, 768), (768, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (3072, 768), (768, 1))
    assert_size_stride(primals_81, (3072, ), (1, ))
    assert_size_stride(primals_82, (768, 3072), (3072, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (2304, 768), (768, 1))
    assert_size_stride(primals_85, (768, 768), (768, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (3072, 768), (768, 1))
    assert_size_stride(primals_88, (3072, ), (1, ))
    assert_size_stride(primals_89, (768, 3072), (3072, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (2304, 768), (768, 1))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (3072, 768), (768, 1))
    assert_size_stride(primals_95, (3072, ), (1, ))
    assert_size_stride(primals_96, (768, 3072), (3072, 1))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (2304, 768), (768, 1))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (3072, 768), (768, 1))
    assert_size_stride(primals_102, (3072, ), (1, ))
    assert_size_stride(primals_103, (768, 3072), (3072, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (2304, 768), (768, 1))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (3072, 768), (768, 1))
    assert_size_stride(primals_109, (3072, ), (1, ))
    assert_size_stride(primals_110, (768, 3072), (3072, 1))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (2304, 768), (768, 1))
    assert_size_stride(primals_113, (768, 768), (768, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (3072, 768), (768, 1))
    assert_size_stride(primals_116, (3072, ), (1, ))
    assert_size_stride(primals_117, (768, 3072), (3072, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (2304, 768), (768, 1))
    assert_size_stride(primals_120, (768, 768), (768, 1))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (3072, 768), (768, 1))
    assert_size_stride(primals_123, (3072, ), (1, ))
    assert_size_stride(primals_124, (768, 3072), (3072, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (2304, 768), (768, 1))
    assert_size_stride(primals_127, (768, 768), (768, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (3072, 768), (768, 1))
    assert_size_stride(primals_130, (3072, ), (1, ))
    assert_size_stride(primals_131, (768, 3072), (3072, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (2304, 768), (768, 1))
    assert_size_stride(primals_134, (768, 768), (768, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (3072, 768), (768, 1))
    assert_size_stride(primals_137, (3072, ), (1, ))
    assert_size_stride(primals_138, (768, 3072), (3072, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (2304, 768), (768, 1))
    assert_size_stride(primals_141, (768, 768), (768, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (3072, 768), (768, 1))
    assert_size_stride(primals_144, (3072, ), (1, ))
    assert_size_stride(primals_145, (768, 3072), (3072, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (2304, 768), (768, 1))
    assert_size_stride(primals_148, (768, 768), (768, 1))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (3072, 768), (768, 1))
    assert_size_stride(primals_151, (3072, ), (1, ))
    assert_size_stride(primals_152, (768, 3072), (3072, 1))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (2304, 768), (768, 1))
    assert_size_stride(primals_155, (768, 768), (768, 1))
    assert_size_stride(primals_156, (768, ), (1, ))
    assert_size_stride(primals_157, (3072, 768), (768, 1))
    assert_size_stride(primals_158, (3072, ), (1, ))
    assert_size_stride(primals_159, (768, 3072), (3072, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (2, 768), (768, 1))
    assert_size_stride(primals_162, (2, ), (1, ))
    assert_size_stride(primals_163, (1, 512), (512, 1))
    assert_size_stride(primals_164, (1, 512), (512, 1))
    assert_size_stride(primals_165, (1, ), (1, ))
    assert_size_stride(primals_166, (1, ), (1, ))
    buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf5 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf25 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf37 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_12, hidden_states_4, query_states], Original ATen: [aten.bernoulli]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_bernoulli_0.run(buf4, buf5, buf25, buf37, 393216, grid=grid(393216), stream=stream0)
        aten.bernoulli_(buf5, 0.9)
        buf1 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf3 = reinterpret_tensor(buf2, (1, 512, 1), (512, 1, 1), 0); del buf2  # reuse
        buf8 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf9 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embeddings, embeddings_1, embeddings_3, hidden_states_1, mean, mul, position_embeddings, pow_1, query_states, sqrt, sub, variance], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub]
        triton_red_fused__to_copy_add_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1.run(buf3, primals_164, primals_75, primals_163, primals_76, buf5, primals_1, primals_2, buf1, buf8, buf9, 512, 768, grid=grid(512), stream=stream0)
        del primals_2
        del primals_75
        del primals_76
        buf10 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (512, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 2304), (1, 768), 0), out=buf10)
        buf11 = reinterpret_tensor(buf5, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf5  # reuse
        buf504 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [query_layer_1, query_layer_2, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf10, primals_3, buf11, buf504, 393216, grid=grid(393216), stream=stream0)
        del primals_3
        buf12 = empty((12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf10, (12, 64, 512), (192, 1, 2304), 64), out=buf12)
        buf15 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf16 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf52 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_1, attention_probs_3], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_3.run(buf15, buf16, buf52, 3145728, grid=grid(3145728), stream=stream0)
        aten.bernoulli_(buf16, 0.9)
        buf19 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf20 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf503 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_1, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf12, buf16, buf19, buf20, buf503, 6144, 512, grid=grid(6144), stream=stream0)
        buf21 = buf11; del buf11  # reuse
        buf502 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_1], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf10, primals_4, buf21, buf502, 393216, grid=grid(393216), stream=stream0)
        del primals_4
        buf22 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf21, (12, 512, 64), (32768, 64, 1), 0), out=buf22)
        buf23 = reinterpret_tensor(buf21, (512, 768), (768, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_3], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf22, buf23, 393216, grid=grid(393216), stream=stream0)
        buf24 = reinterpret_tensor(buf22, (512, 768), (768, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf23, reinterpret_tensor(primals_78, (768, 768), (1, 768), 0), out=buf24)
        aten.bernoulli_(buf25, 0.9)
        buf28 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf30 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf32 = reinterpret_tensor(buf31, (1, 512, 1), (512, 1, 1), 0); del buf31  # reuse
        buf33 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_4, add_5, attention_output, hidden_states_4, hidden_states_6, hidden_states_9, mean_3, mul_4, pow_2, query_states, sqrt_2, sub_2, variance_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_7.run(buf32, buf25, buf24, primals_79, buf9, primals_5, primals_6, buf28, buf30, buf33, 512, 768, grid=grid(512), stream=stream0)
        del primals_79
        buf34 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf33, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf34)
        del primals_81
        buf35 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_11, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf34, buf35, 1572864, grid=grid(1572864), stream=stream0)
        buf36 = reinterpret_tensor(buf25, (512, 768), (768, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf35, reinterpret_tensor(primals_82, (3072, 768), (1, 3072), 0), out=buf36)
        aten.bernoulli_(buf37, 0.9)
        buf40 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf41 = reinterpret_tensor(buf36, (1, 512, 768), (393216, 768, 1), 0); del buf36  # reuse
        buf43 = reinterpret_tensor(buf24, (1, 512, 768), (393216, 768, 1), 0); del buf24  # reuse
        buf44 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf45 = reinterpret_tensor(buf44, (1, 512, 1), (512, 1, 1), 0); del buf44  # reuse
        buf46 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_7, add_8, attention_output, hidden_states_12, hidden_states_14, hidden_states_6, mean_6, mul_4, mul_5, pow_3, qp_1, query_states, query_states_1, sqrt_3, sub_4, variance_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf41, buf45, buf37, primals_83, primals_5, buf30, buf32, primals_6, primals_7, primals_8, buf40, buf43, buf46, 512, 768, grid=grid(512), stream=stream0)
        del primals_6
        del primals_83
        buf47 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_84, (768, 2304), (1, 768), 0), out=buf47)
        buf48 = reinterpret_tensor(buf41, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf41  # reuse
        buf501 = reinterpret_tensor(buf37, (12, 64, 512), (64, 1, 768), 0); del buf37  # reuse
        # Source Nodes: [query_layer_4, query_layer_5, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf47, primals_9, buf48, buf501, 393216, grid=grid(393216), stream=stream0)
        del primals_9
        buf49 = reinterpret_tensor(buf16, (12, 512, 512), (262144, 512, 1), 0); del buf16  # reuse
        # Source Nodes: [attention_scores_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf47, (12, 64, 512), (192, 1, 2304), 64), out=buf49)
        aten.bernoulli_(buf52, 0.9)
        buf55 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf56 = reinterpret_tensor(buf12, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf12  # reuse
        buf500 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_2, attention_probs_3, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf49, buf52, buf55, buf56, buf500, 6144, 512, grid=grid(6144), stream=stream0)
        buf57 = buf48; del buf48  # reuse
        buf499 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_3], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf47, primals_10, buf57, buf499, 393216, grid=grid(393216), stream=stream0)
        del primals_10
        buf58 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf56, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf57, (12, 512, 64), (32768, 64, 1), 0), out=buf58)
        buf59 = reinterpret_tensor(buf57, (512, 768), (768, 1), 0); del buf57  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf58, buf59, 393216, grid=grid(393216), stream=stream0)
        buf60 = reinterpret_tensor(buf58, (512, 768), (768, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf59, reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), out=buf60)
        buf61 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf74 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf98 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf111 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_19, hidden_states_27, hidden_states_34, hidden_states_42], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_10.run(buf4, buf61, buf74, buf98, buf111, 393216, grid=grid(393216), stream=stream0)
        aten.bernoulli_(buf61, 0.9)
        buf64 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf65 = reinterpret_tensor(buf60, (1, 512, 768), (393216, 768, 1), 0); del buf60  # reuse
        buf67 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf69 = reinterpret_tensor(buf68, (1, 512, 1), (512, 1, 1), 0); del buf68  # reuse
        buf70 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, add_13, attention_output_2, hidden_states_14, hidden_states_19, hidden_states_21, hidden_states_24, mean_9, mul_5, mul_7, pow_4, query_states, query_states_1, sqrt_5, sub_6, variance_3], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf65, buf69, buf61, primals_86, primals_7, buf43, buf45, primals_8, primals_11, primals_12, buf64, buf67, buf70, 512, 768, grid=grid(512), stream=stream0)
        del primals_8
        del primals_86
        buf71 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_88, buf70, reinterpret_tensor(primals_87, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf71)
        del primals_88
        buf72 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_26, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf71, buf72, 1572864, grid=grid(1572864), stream=stream0)
        buf73 = reinterpret_tensor(buf65, (512, 768), (768, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf72, reinterpret_tensor(primals_89, (3072, 768), (1, 3072), 0), out=buf73)
        aten.bernoulli_(buf74, 0.9)
        buf77 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf78 = reinterpret_tensor(buf73, (1, 512, 768), (393216, 768, 1), 0); del buf73  # reuse
        buf80 = buf61; del buf61  # reuse
        buf81 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf82 = reinterpret_tensor(buf81, (1, 512, 1), (512, 1, 1), 0); del buf81  # reuse
        buf83 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, add_16, attention_output_2, hidden_states_21, hidden_states_27, hidden_states_29, mean_12, mul_7, mul_8, pow_5, qp_2, query_states, query_states_2, sqrt_6, sub_8, variance_4], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf78, buf82, buf74, primals_90, primals_11, buf67, buf69, primals_12, primals_13, primals_14, buf77, buf80, buf83, 512, 768, grid=grid(512), stream=stream0)
        del primals_12
        del primals_90
        buf84 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf83, reinterpret_tensor(primals_91, (768, 2304), (1, 768), 0), out=buf84)
        buf85 = reinterpret_tensor(buf78, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf78  # reuse
        buf498 = reinterpret_tensor(buf74, (12, 64, 512), (64, 1, 768), 0); del buf74  # reuse
        # Source Nodes: [query_layer_7, query_layer_8, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf84, primals_15, buf85, buf498, 393216, grid=grid(393216), stream=stream0)
        del primals_15
        buf86 = reinterpret_tensor(buf52, (12, 512, 512), (262144, 512, 1), 0); del buf52  # reuse
        # Source Nodes: [attention_scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf84, (12, 64, 512), (192, 1, 2304), 64), out=buf86)
        buf89 = reinterpret_tensor(buf49, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf49  # reuse
        buf126 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_5, attention_probs_7], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_3.run(buf15, buf89, buf126, 3145728, grid=grid(3145728), stream=stream0)
        aten.bernoulli_(buf89, 0.9)
        buf92 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf93 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf497 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_4, attention_probs_5, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf86, buf89, buf92, buf93, buf497, 6144, 512, grid=grid(6144), stream=stream0)
        buf94 = buf85; del buf85  # reuse
        buf496 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_5], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf84, primals_16, buf94, buf496, 393216, grid=grid(393216), stream=stream0)
        del primals_16
        buf95 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf94, (12, 512, 64), (32768, 64, 1), 0), out=buf95)
        buf96 = reinterpret_tensor(buf94, (512, 768), (768, 1), 0); del buf94  # reuse
        # Source Nodes: [hidden_states_33], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf95, buf96, 393216, grid=grid(393216), stream=stream0)
        buf97 = reinterpret_tensor(buf95, (512, 768), (768, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf96, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), out=buf97)
        aten.bernoulli_(buf98, 0.9)
        buf101 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf102 = reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0); del buf97  # reuse
        buf104 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf105 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf106 = reinterpret_tensor(buf105, (1, 512, 1), (512, 1, 1), 0); del buf105  # reuse
        buf107 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, add_21, attention_output_4, hidden_states_29, hidden_states_34, hidden_states_36, hidden_states_39, mean_15, mul_10, mul_8, pow_6, query_states, query_states_2, sqrt_8, sub_10, variance_5], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf102, buf106, buf98, primals_93, primals_13, buf80, buf82, primals_14, primals_17, primals_18, buf101, buf104, buf107, 512, 768, grid=grid(512), stream=stream0)
        del primals_14
        del primals_93
        buf108 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_95, buf107, reinterpret_tensor(primals_94, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf108)
        del primals_95
        buf109 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_41, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf108, buf109, 1572864, grid=grid(1572864), stream=stream0)
        buf110 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf109, reinterpret_tensor(primals_96, (3072, 768), (1, 3072), 0), out=buf110)
        aten.bernoulli_(buf111, 0.9)
        buf114 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf115 = reinterpret_tensor(buf110, (1, 512, 768), (393216, 768, 1), 0); del buf110  # reuse
        buf117 = buf102; del buf102  # reuse
        buf118 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf119 = reinterpret_tensor(buf118, (1, 512, 1), (512, 1, 1), 0); del buf118  # reuse
        buf120 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, add_24, attention_output_4, hidden_states_36, hidden_states_42, hidden_states_44, mean_18, mul_10, mul_11, pow_7, qp_3, query_states, query_states_3, sqrt_9, sub_12, variance_6], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf115, buf119, buf111, primals_97, primals_17, buf104, buf106, primals_18, primals_19, primals_20, buf114, buf117, buf120, 512, 768, grid=grid(512), stream=stream0)
        del primals_18
        del primals_97
        buf121 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf120, reinterpret_tensor(primals_98, (768, 2304), (1, 768), 0), out=buf121)
        buf122 = reinterpret_tensor(buf115, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf115  # reuse
        buf495 = reinterpret_tensor(buf111, (12, 64, 512), (64, 1, 768), 0); del buf111  # reuse
        # Source Nodes: [query_layer_10, query_layer_11, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf121, primals_21, buf122, buf495, 393216, grid=grid(393216), stream=stream0)
        del primals_21
        buf123 = reinterpret_tensor(buf89, (12, 512, 512), (262144, 512, 1), 0); del buf89  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf121, (12, 64, 512), (192, 1, 2304), 64), out=buf123)
        aten.bernoulli_(buf126, 0.9)
        buf129 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf130 = reinterpret_tensor(buf86, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf86  # reuse
        buf494 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_6, attention_probs_7, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf123, buf126, buf129, buf130, buf494, 6144, 512, grid=grid(6144), stream=stream0)
        buf131 = buf122; del buf122  # reuse
        buf493 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_7], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf121, primals_22, buf131, buf493, 393216, grid=grid(393216), stream=stream0)
        del primals_22
        buf132 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf131, (12, 512, 64), (32768, 64, 1), 0), out=buf132)
        buf133 = reinterpret_tensor(buf131, (512, 768), (768, 1), 0); del buf131  # reuse
        # Source Nodes: [hidden_states_48], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf132, buf133, 393216, grid=grid(393216), stream=stream0)
        buf134 = reinterpret_tensor(buf132, (512, 768), (768, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf133, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), out=buf134)
        buf135 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf148 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf172 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf185 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_49, hidden_states_57, hidden_states_64, hidden_states_72], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_10.run(buf4, buf135, buf148, buf172, buf185, 393216, grid=grid(393216), stream=stream0)
        aten.bernoulli_(buf135, 0.9)
        buf138 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf139 = reinterpret_tensor(buf134, (1, 512, 768), (393216, 768, 1), 0); del buf134  # reuse
        buf141 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf143 = reinterpret_tensor(buf142, (1, 512, 1), (512, 1, 1), 0); del buf142  # reuse
        buf144 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, add_29, attention_output_6, hidden_states_44, hidden_states_49, hidden_states_51, hidden_states_54, mean_21, mul_11, mul_13, pow_8, query_states, query_states_3, sqrt_11, sub_14, variance_7], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf139, buf143, buf135, primals_100, primals_19, buf117, buf119, primals_20, primals_23, primals_24, buf138, buf141, buf144, 512, 768, grid=grid(512), stream=stream0)
        del primals_100
        del primals_20
        buf145 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, buf144, reinterpret_tensor(primals_101, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf145)
        del primals_102
        buf146 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_56, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf145, buf146, 1572864, grid=grid(1572864), stream=stream0)
        buf147 = reinterpret_tensor(buf139, (512, 768), (768, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf146, reinterpret_tensor(primals_103, (3072, 768), (1, 3072), 0), out=buf147)
        aten.bernoulli_(buf148, 0.9)
        buf151 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf152 = reinterpret_tensor(buf147, (1, 512, 768), (393216, 768, 1), 0); del buf147  # reuse
        buf154 = buf135; del buf135  # reuse
        buf155 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf156 = reinterpret_tensor(buf155, (1, 512, 1), (512, 1, 1), 0); del buf155  # reuse
        buf157 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_31, add_32, attention_output_6, hidden_states_51, hidden_states_57, hidden_states_59, mean_24, mul_13, mul_14, pow_9, qp_4, query_states, query_states_4, sqrt_12, sub_16, variance_8], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf152, buf156, buf148, primals_104, primals_23, buf141, buf143, primals_24, primals_25, primals_26, buf151, buf154, buf157, 512, 768, grid=grid(512), stream=stream0)
        del primals_104
        del primals_24
        buf158 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf157, reinterpret_tensor(primals_105, (768, 2304), (1, 768), 0), out=buf158)
        buf159 = reinterpret_tensor(buf152, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf152  # reuse
        buf492 = reinterpret_tensor(buf148, (12, 64, 512), (64, 1, 768), 0); del buf148  # reuse
        # Source Nodes: [query_layer_13, query_layer_14, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf158, primals_27, buf159, buf492, 393216, grid=grid(393216), stream=stream0)
        del primals_27
        buf160 = reinterpret_tensor(buf126, (12, 512, 512), (262144, 512, 1), 0); del buf126  # reuse
        # Source Nodes: [attention_scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf159, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf158, (12, 64, 512), (192, 1, 2304), 64), out=buf160)
        buf163 = reinterpret_tensor(buf123, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf123  # reuse
        buf200 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_11, attention_probs_9], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_3.run(buf15, buf163, buf200, 3145728, grid=grid(3145728), stream=stream0)
        aten.bernoulli_(buf163, 0.9)
        buf166 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf167 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf491 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_8, attention_probs_9, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf160, buf163, buf166, buf167, buf491, 6144, 512, grid=grid(6144), stream=stream0)
        buf168 = buf159; del buf159  # reuse
        buf490 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_9], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf158, primals_28, buf168, buf490, 393216, grid=grid(393216), stream=stream0)
        del primals_28
        buf169 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf168, (12, 512, 64), (32768, 64, 1), 0), out=buf169)
        buf170 = reinterpret_tensor(buf168, (512, 768), (768, 1), 0); del buf168  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf169, buf170, 393216, grid=grid(393216), stream=stream0)
        buf171 = reinterpret_tensor(buf169, (512, 768), (768, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf170, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), out=buf171)
        aten.bernoulli_(buf172, 0.9)
        buf175 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf176 = reinterpret_tensor(buf171, (1, 512, 768), (393216, 768, 1), 0); del buf171  # reuse
        buf178 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf179 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf180 = reinterpret_tensor(buf179, (1, 512, 1), (512, 1, 1), 0); del buf179  # reuse
        buf181 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, add_37, attention_output_8, hidden_states_59, hidden_states_64, hidden_states_66, hidden_states_69, mean_27, mul_14, mul_16, pow_10, query_states, query_states_4, sqrt_14, sub_18, variance_9], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf176, buf180, buf172, primals_107, primals_25, buf154, buf156, primals_26, primals_29, primals_30, buf175, buf178, buf181, 512, 768, grid=grid(512), stream=stream0)
        del primals_107
        del primals_26
        buf182 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_109, buf181, reinterpret_tensor(primals_108, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf182)
        del primals_109
        buf183 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_71, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf182, buf183, 1572864, grid=grid(1572864), stream=stream0)
        buf184 = reinterpret_tensor(buf176, (512, 768), (768, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf183, reinterpret_tensor(primals_110, (3072, 768), (1, 3072), 0), out=buf184)
        aten.bernoulli_(buf185, 0.9)
        buf188 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf189 = reinterpret_tensor(buf184, (1, 512, 768), (393216, 768, 1), 0); del buf184  # reuse
        buf191 = buf172; del buf172  # reuse
        buf192 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf193 = reinterpret_tensor(buf192, (1, 512, 1), (512, 1, 1), 0); del buf192  # reuse
        buf194 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_39, add_40, attention_output_8, hidden_states_66, hidden_states_72, hidden_states_74, mean_30, mul_16, mul_17, pow_11, qp_5, query_states, query_states_5, sqrt_15, sub_20, variance_10], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf189, buf193, buf185, primals_111, primals_29, buf178, buf180, primals_30, primals_31, primals_32, buf188, buf191, buf194, 512, 768, grid=grid(512), stream=stream0)
        del primals_111
        del primals_30
        buf195 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf194, reinterpret_tensor(primals_112, (768, 2304), (1, 768), 0), out=buf195)
        buf196 = reinterpret_tensor(buf189, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf189  # reuse
        buf489 = reinterpret_tensor(buf185, (12, 64, 512), (64, 1, 768), 0); del buf185  # reuse
        # Source Nodes: [query_layer_16, query_layer_17, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf195, primals_33, buf196, buf489, 393216, grid=grid(393216), stream=stream0)
        del primals_33
        buf197 = reinterpret_tensor(buf163, (12, 512, 512), (262144, 512, 1), 0); del buf163  # reuse
        # Source Nodes: [attention_scores_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf196, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf195, (12, 64, 512), (192, 1, 2304), 64), out=buf197)
        aten.bernoulli_(buf200, 0.9)
        buf203 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf204 = reinterpret_tensor(buf160, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf160  # reuse
        buf488 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_10, attention_probs_11, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf197, buf200, buf203, buf204, buf488, 6144, 512, grid=grid(6144), stream=stream0)
        buf205 = buf196; del buf196  # reuse
        buf487 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_11], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf195, primals_34, buf205, buf487, 393216, grid=grid(393216), stream=stream0)
        del primals_34
        buf206 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf205, (12, 512, 64), (32768, 64, 1), 0), out=buf206)
        buf207 = reinterpret_tensor(buf205, (512, 768), (768, 1), 0); del buf205  # reuse
        # Source Nodes: [hidden_states_78], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf206, buf207, 393216, grid=grid(393216), stream=stream0)
        buf208 = reinterpret_tensor(buf206, (512, 768), (768, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf207, reinterpret_tensor(primals_113, (768, 768), (1, 768), 0), out=buf208)
        buf209 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf222 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf246 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf259 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102, hidden_states_79, hidden_states_87, hidden_states_94], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_10.run(buf4, buf209, buf222, buf246, buf259, 393216, grid=grid(393216), stream=stream0)
        aten.bernoulli_(buf209, 0.9)
        buf212 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf213 = reinterpret_tensor(buf208, (1, 512, 768), (393216, 768, 1), 0); del buf208  # reuse
        buf215 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf216 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf217 = reinterpret_tensor(buf216, (1, 512, 1), (512, 1, 1), 0); del buf216  # reuse
        buf218 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, add_45, attention_output_10, hidden_states_74, hidden_states_79, hidden_states_81, hidden_states_84, mean_33, mul_17, mul_19, pow_12, query_states, query_states_5, sqrt_17, sub_22, variance_11], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf213, buf217, buf209, primals_114, primals_31, buf191, buf193, primals_32, primals_35, primals_36, buf212, buf215, buf218, 512, 768, grid=grid(512), stream=stream0)
        del primals_114
        del primals_32
        buf219 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf218, reinterpret_tensor(primals_115, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf219)
        del primals_116
        buf220 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_86, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf219, buf220, 1572864, grid=grid(1572864), stream=stream0)
        buf221 = reinterpret_tensor(buf213, (512, 768), (768, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf220, reinterpret_tensor(primals_117, (3072, 768), (1, 3072), 0), out=buf221)
        aten.bernoulli_(buf222, 0.9)
        buf225 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf226 = reinterpret_tensor(buf221, (1, 512, 768), (393216, 768, 1), 0); del buf221  # reuse
        buf228 = buf209; del buf209  # reuse
        buf229 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf230 = reinterpret_tensor(buf229, (1, 512, 1), (512, 1, 1), 0); del buf229  # reuse
        buf231 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_47, add_48, attention_output_10, hidden_states_81, hidden_states_87, hidden_states_89, mean_36, mul_19, mul_20, pow_13, qp_6, query_states, query_states_6, sqrt_18, sub_24, variance_12], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf226, buf230, buf222, primals_118, primals_35, buf215, buf217, primals_36, primals_37, primals_38, buf225, buf228, buf231, 512, 768, grid=grid(512), stream=stream0)
        del primals_118
        del primals_36
        buf232 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf231, reinterpret_tensor(primals_119, (768, 2304), (1, 768), 0), out=buf232)
        buf233 = reinterpret_tensor(buf226, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf226  # reuse
        buf486 = reinterpret_tensor(buf222, (12, 64, 512), (64, 1, 768), 0); del buf222  # reuse
        # Source Nodes: [query_layer_19, query_layer_20, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf232, primals_39, buf233, buf486, 393216, grid=grid(393216), stream=stream0)
        del primals_39
        buf234 = reinterpret_tensor(buf200, (12, 512, 512), (262144, 512, 1), 0); del buf200  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf232, (12, 64, 512), (192, 1, 2304), 64), out=buf234)
        buf237 = reinterpret_tensor(buf197, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf197  # reuse
        buf274 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_13, attention_probs_15], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_3.run(buf15, buf237, buf274, 3145728, grid=grid(3145728), stream=stream0)
        aten.bernoulli_(buf237, 0.9)
        buf240 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf241 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf485 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_12, attention_probs_13, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf234, buf237, buf240, buf241, buf485, 6144, 512, grid=grid(6144), stream=stream0)
        buf242 = buf233; del buf233  # reuse
        buf484 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_13], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf232, primals_40, buf242, buf484, 393216, grid=grid(393216), stream=stream0)
        del primals_40
        buf243 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf242, (12, 512, 64), (32768, 64, 1), 0), out=buf243)
        buf244 = reinterpret_tensor(buf242, (512, 768), (768, 1), 0); del buf242  # reuse
        # Source Nodes: [hidden_states_93], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf243, buf244, 393216, grid=grid(393216), stream=stream0)
        buf245 = reinterpret_tensor(buf243, (512, 768), (768, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf244, reinterpret_tensor(primals_120, (768, 768), (1, 768), 0), out=buf245)
        aten.bernoulli_(buf246, 0.9)
        buf249 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf250 = reinterpret_tensor(buf245, (1, 512, 768), (393216, 768, 1), 0); del buf245  # reuse
        buf252 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf253 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf254 = reinterpret_tensor(buf253, (1, 512, 1), (512, 1, 1), 0); del buf253  # reuse
        buf255 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_52, add_53, attention_output_12, hidden_states_89, hidden_states_94, hidden_states_96, hidden_states_99, mean_39, mul_20, mul_22, pow_14, query_states, query_states_6, sqrt_20, sub_26, variance_13], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf250, buf254, buf246, primals_121, primals_37, buf228, buf230, primals_38, primals_41, primals_42, buf249, buf252, buf255, 512, 768, grid=grid(512), stream=stream0)
        del primals_121
        del primals_38
        buf256 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_123, buf255, reinterpret_tensor(primals_122, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf256)
        del primals_123
        buf257 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_101, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf256, buf257, 1572864, grid=grid(1572864), stream=stream0)
        buf258 = reinterpret_tensor(buf250, (512, 768), (768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf257, reinterpret_tensor(primals_124, (3072, 768), (1, 3072), 0), out=buf258)
        aten.bernoulli_(buf259, 0.9)
        buf262 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf263 = reinterpret_tensor(buf258, (1, 512, 768), (393216, 768, 1), 0); del buf258  # reuse
        buf265 = buf246; del buf246  # reuse
        buf266 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf267 = reinterpret_tensor(buf266, (1, 512, 1), (512, 1, 1), 0); del buf266  # reuse
        buf268 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_55, add_56, attention_output_12, hidden_states_102, hidden_states_104, hidden_states_96, mean_42, mul_22, mul_23, pow_15, qp_7, query_states, query_states_7, sqrt_21, sub_28, variance_14], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf263, buf267, buf259, primals_125, primals_41, buf252, buf254, primals_42, primals_43, primals_44, buf262, buf265, buf268, 512, 768, grid=grid(512), stream=stream0)
        del primals_125
        del primals_42
        buf269 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf268, reinterpret_tensor(primals_126, (768, 2304), (1, 768), 0), out=buf269)
        buf270 = reinterpret_tensor(buf263, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf263  # reuse
        buf483 = reinterpret_tensor(buf259, (12, 64, 512), (64, 1, 768), 0); del buf259  # reuse
        # Source Nodes: [query_layer_22, query_layer_23, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf269, primals_45, buf270, buf483, 393216, grid=grid(393216), stream=stream0)
        del primals_45
        buf271 = reinterpret_tensor(buf237, (12, 512, 512), (262144, 512, 1), 0); del buf237  # reuse
        # Source Nodes: [attention_scores_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf269, (12, 64, 512), (192, 1, 2304), 64), out=buf271)
        aten.bernoulli_(buf274, 0.9)
        buf277 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf278 = reinterpret_tensor(buf234, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf234  # reuse
        buf482 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_14, attention_probs_15, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf271, buf274, buf277, buf278, buf482, 6144, 512, grid=grid(6144), stream=stream0)
        buf279 = buf270; del buf270  # reuse
        buf481 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_15], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf269, primals_46, buf279, buf481, 393216, grid=grid(393216), stream=stream0)
        del primals_46
        buf280 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf279, (12, 512, 64), (32768, 64, 1), 0), out=buf280)
        buf281 = reinterpret_tensor(buf279, (512, 768), (768, 1), 0); del buf279  # reuse
        # Source Nodes: [hidden_states_108], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf280, buf281, 393216, grid=grid(393216), stream=stream0)
        buf282 = reinterpret_tensor(buf280, (512, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf281, reinterpret_tensor(primals_127, (768, 768), (1, 768), 0), out=buf282)
        buf283 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf296 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf320 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf333 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_117, hidden_states_124, hidden_states_132], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_10.run(buf4, buf283, buf296, buf320, buf333, 393216, grid=grid(393216), stream=stream0)
        aten.bernoulli_(buf283, 0.9)
        buf286 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf287 = reinterpret_tensor(buf282, (1, 512, 768), (393216, 768, 1), 0); del buf282  # reuse
        buf289 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf290 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf291 = reinterpret_tensor(buf290, (1, 512, 1), (512, 1, 1), 0); del buf290  # reuse
        buf292 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_60, add_61, attention_output_14, hidden_states_104, hidden_states_109, hidden_states_111, hidden_states_114, mean_45, mul_23, mul_25, pow_16, query_states, query_states_7, sqrt_23, sub_30, variance_15], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf287, buf291, buf283, primals_128, primals_43, buf265, buf267, primals_44, primals_47, primals_48, buf286, buf289, buf292, 512, 768, grid=grid(512), stream=stream0)
        del primals_128
        del primals_44
        buf293 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_130, buf292, reinterpret_tensor(primals_129, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf293)
        del primals_130
        buf294 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_116, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf293, buf294, 1572864, grid=grid(1572864), stream=stream0)
        buf295 = reinterpret_tensor(buf287, (512, 768), (768, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf294, reinterpret_tensor(primals_131, (3072, 768), (1, 3072), 0), out=buf295)
        aten.bernoulli_(buf296, 0.9)
        buf299 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf300 = reinterpret_tensor(buf295, (1, 512, 768), (393216, 768, 1), 0); del buf295  # reuse
        buf302 = buf283; del buf283  # reuse
        buf303 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf304 = reinterpret_tensor(buf303, (1, 512, 1), (512, 1, 1), 0); del buf303  # reuse
        buf305 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_63, add_64, attention_output_14, hidden_states_111, hidden_states_117, hidden_states_119, mean_48, mul_25, mul_26, pow_17, qp_8, query_states, query_states_8, sqrt_24, sub_32, variance_16], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf300, buf304, buf296, primals_132, primals_47, buf289, buf291, primals_48, primals_49, primals_50, buf299, buf302, buf305, 512, 768, grid=grid(512), stream=stream0)
        del primals_132
        del primals_48
        buf306 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf305, reinterpret_tensor(primals_133, (768, 2304), (1, 768), 0), out=buf306)
        buf307 = reinterpret_tensor(buf300, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf300  # reuse
        buf480 = reinterpret_tensor(buf296, (12, 64, 512), (64, 1, 768), 0); del buf296  # reuse
        # Source Nodes: [query_layer_25, query_layer_26, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf306, primals_51, buf307, buf480, 393216, grid=grid(393216), stream=stream0)
        del primals_51
        buf308 = reinterpret_tensor(buf274, (12, 512, 512), (262144, 512, 1), 0); del buf274  # reuse
        # Source Nodes: [attention_scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf306, (12, 64, 512), (192, 1, 2304), 64), out=buf308)
        buf311 = reinterpret_tensor(buf271, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf271  # reuse
        buf348 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_17, attention_probs_19], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_3.run(buf15, buf311, buf348, 3145728, grid=grid(3145728), stream=stream0)
        aten.bernoulli_(buf311, 0.9)
        buf314 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf315 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf479 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_16, attention_probs_17, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf308, buf311, buf314, buf315, buf479, 6144, 512, grid=grid(6144), stream=stream0)
        buf316 = buf307; del buf307  # reuse
        buf478 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_17], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf306, primals_52, buf316, buf478, 393216, grid=grid(393216), stream=stream0)
        del primals_52
        buf317 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf316, (12, 512, 64), (32768, 64, 1), 0), out=buf317)
        buf318 = reinterpret_tensor(buf316, (512, 768), (768, 1), 0); del buf316  # reuse
        # Source Nodes: [hidden_states_123], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf317, buf318, 393216, grid=grid(393216), stream=stream0)
        buf319 = reinterpret_tensor(buf317, (512, 768), (768, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf318, reinterpret_tensor(primals_134, (768, 768), (1, 768), 0), out=buf319)
        aten.bernoulli_(buf320, 0.9)
        buf323 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf324 = reinterpret_tensor(buf319, (1, 512, 768), (393216, 768, 1), 0); del buf319  # reuse
        buf326 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf327 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf328 = reinterpret_tensor(buf327, (1, 512, 1), (512, 1, 1), 0); del buf327  # reuse
        buf329 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, add_69, attention_output_16, hidden_states_119, hidden_states_124, hidden_states_126, hidden_states_129, mean_51, mul_26, mul_28, pow_18, query_states, query_states_8, sqrt_26, sub_34, variance_17], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf324, buf328, buf320, primals_135, primals_49, buf302, buf304, primals_50, primals_53, primals_54, buf323, buf326, buf329, 512, 768, grid=grid(512), stream=stream0)
        del primals_135
        del primals_50
        buf330 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_129], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_137, buf329, reinterpret_tensor(primals_136, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf330)
        del primals_137
        buf331 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_131, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf330, buf331, 1572864, grid=grid(1572864), stream=stream0)
        buf332 = reinterpret_tensor(buf324, (512, 768), (768, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf331, reinterpret_tensor(primals_138, (3072, 768), (1, 3072), 0), out=buf332)
        aten.bernoulli_(buf333, 0.9)
        buf336 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf337 = reinterpret_tensor(buf332, (1, 512, 768), (393216, 768, 1), 0); del buf332  # reuse
        buf339 = buf320; del buf320  # reuse
        buf340 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf341 = reinterpret_tensor(buf340, (1, 512, 1), (512, 1, 1), 0); del buf340  # reuse
        buf342 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_71, add_72, attention_output_16, hidden_states_126, hidden_states_132, hidden_states_134, mean_54, mul_28, mul_29, pow_19, qp_9, query_states, query_states_9, sqrt_27, sub_36, variance_18], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf337, buf341, buf333, primals_139, primals_53, buf326, buf328, primals_54, primals_55, primals_56, buf336, buf339, buf342, 512, 768, grid=grid(512), stream=stream0)
        del primals_139
        del primals_54
        buf343 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf342, reinterpret_tensor(primals_140, (768, 2304), (1, 768), 0), out=buf343)
        buf344 = reinterpret_tensor(buf337, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf337  # reuse
        buf477 = reinterpret_tensor(buf333, (12, 64, 512), (64, 1, 768), 0); del buf333  # reuse
        # Source Nodes: [query_layer_28, query_layer_29, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf343, primals_57, buf344, buf477, 393216, grid=grid(393216), stream=stream0)
        del primals_57
        buf345 = reinterpret_tensor(buf311, (12, 512, 512), (262144, 512, 1), 0); del buf311  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf344, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf343, (12, 64, 512), (192, 1, 2304), 64), out=buf345)
        aten.bernoulli_(buf348, 0.9)
        buf351 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf352 = reinterpret_tensor(buf308, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf308  # reuse
        buf476 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_18, attention_probs_19, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf345, buf348, buf351, buf352, buf476, 6144, 512, grid=grid(6144), stream=stream0)
        buf353 = buf344; del buf344  # reuse
        buf475 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_19], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf343, primals_58, buf353, buf475, 393216, grid=grid(393216), stream=stream0)
        del primals_58
        buf354 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf352, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf353, (12, 512, 64), (32768, 64, 1), 0), out=buf354)
        buf355 = reinterpret_tensor(buf353, (512, 768), (768, 1), 0); del buf353  # reuse
        # Source Nodes: [hidden_states_138], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf354, buf355, 393216, grid=grid(393216), stream=stream0)
        buf356 = reinterpret_tensor(buf354, (512, 768), (768, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf355, reinterpret_tensor(primals_141, (768, 768), (1, 768), 0), out=buf356)
        buf357 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf370 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf394 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf407 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_139, hidden_states_147, hidden_states_154, hidden_states_162], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_10.run(buf4, buf357, buf370, buf394, buf407, 393216, grid=grid(393216), stream=stream0)
        aten.bernoulli_(buf357, 0.9)
        buf360 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf361 = reinterpret_tensor(buf356, (1, 512, 768), (393216, 768, 1), 0); del buf356  # reuse
        buf363 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf364 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf365 = reinterpret_tensor(buf364, (1, 512, 1), (512, 1, 1), 0); del buf364  # reuse
        buf366 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_76, add_77, attention_output_18, hidden_states_134, hidden_states_139, hidden_states_141, hidden_states_144, mean_57, mul_29, mul_31, pow_20, query_states, query_states_9, sqrt_29, sub_38, variance_19], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf361, buf365, buf357, primals_142, primals_55, buf339, buf341, primals_56, primals_59, primals_60, buf360, buf363, buf366, 512, 768, grid=grid(512), stream=stream0)
        del primals_142
        del primals_56
        buf367 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf366, reinterpret_tensor(primals_143, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf367)
        del primals_144
        buf368 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_146, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf367, buf368, 1572864, grid=grid(1572864), stream=stream0)
        buf369 = reinterpret_tensor(buf361, (512, 768), (768, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf368, reinterpret_tensor(primals_145, (3072, 768), (1, 3072), 0), out=buf369)
        aten.bernoulli_(buf370, 0.9)
        buf373 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf374 = reinterpret_tensor(buf369, (1, 512, 768), (393216, 768, 1), 0); del buf369  # reuse
        buf376 = buf357; del buf357  # reuse
        buf377 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf378 = reinterpret_tensor(buf377, (1, 512, 1), (512, 1, 1), 0); del buf377  # reuse
        buf379 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_79, add_80, attention_output_18, hidden_states_141, hidden_states_147, hidden_states_149, mean_60, mul_31, mul_32, pow_21, qp_10, query_states, query_states_10, sqrt_30, sub_40, variance_20], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf374, buf378, buf370, primals_146, primals_59, buf363, buf365, primals_60, primals_61, primals_62, buf373, buf376, buf379, 512, 768, grid=grid(512), stream=stream0)
        del primals_146
        del primals_60
        buf380 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf379, reinterpret_tensor(primals_147, (768, 2304), (1, 768), 0), out=buf380)
        buf381 = reinterpret_tensor(buf374, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf374  # reuse
        buf474 = reinterpret_tensor(buf370, (12, 64, 512), (64, 1, 768), 0); del buf370  # reuse
        # Source Nodes: [query_layer_31, query_layer_32, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf380, primals_63, buf381, buf474, 393216, grid=grid(393216), stream=stream0)
        del primals_63
        buf382 = reinterpret_tensor(buf348, (12, 512, 512), (262144, 512, 1), 0); del buf348  # reuse
        # Source Nodes: [attention_scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf381, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf380, (12, 64, 512), (192, 1, 2304), 64), out=buf382)
        buf385 = reinterpret_tensor(buf345, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf345  # reuse
        buf422 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_21, attention_probs_23], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_3.run(buf15, buf385, buf422, 3145728, grid=grid(3145728), stream=stream0)
        aten.bernoulli_(buf385, 0.9)
        buf388 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf389 = buf15; del buf15  # reuse
        buf473 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_20, attention_probs_21, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf382, buf385, buf388, buf389, buf473, 6144, 512, grid=grid(6144), stream=stream0)
        buf390 = buf381; del buf381  # reuse
        buf472 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_21], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf380, primals_64, buf390, buf472, 393216, grid=grid(393216), stream=stream0)
        del primals_64
        buf391 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf389, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf390, (12, 512, 64), (32768, 64, 1), 0), out=buf391)
        buf392 = reinterpret_tensor(buf390, (512, 768), (768, 1), 0); del buf390  # reuse
        # Source Nodes: [hidden_states_153], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf391, buf392, 393216, grid=grid(393216), stream=stream0)
        buf393 = reinterpret_tensor(buf391, (512, 768), (768, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf392, reinterpret_tensor(primals_148, (768, 768), (1, 768), 0), out=buf393)
        aten.bernoulli_(buf394, 0.9)
        buf397 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf398 = reinterpret_tensor(buf393, (1, 512, 768), (393216, 768, 1), 0); del buf393  # reuse
        buf400 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf401 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf402 = reinterpret_tensor(buf401, (1, 512, 1), (512, 1, 1), 0); del buf401  # reuse
        buf403 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_84, add_85, attention_output_20, hidden_states_149, hidden_states_154, hidden_states_156, hidden_states_159, mean_63, mul_32, mul_34, pow_22, query_states, query_states_10, sqrt_32, sub_42, variance_21], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf398, buf402, buf394, primals_149, primals_61, buf376, buf378, primals_62, primals_65, primals_66, buf397, buf400, buf403, 512, 768, grid=grid(512), stream=stream0)
        del primals_149
        del primals_62
        buf404 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_151, buf403, reinterpret_tensor(primals_150, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf404)
        del primals_151
        buf405 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_161, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf404, buf405, 1572864, grid=grid(1572864), stream=stream0)
        buf406 = reinterpret_tensor(buf398, (512, 768), (768, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf405, reinterpret_tensor(primals_152, (3072, 768), (1, 3072), 0), out=buf406)
        aten.bernoulli_(buf407, 0.9)
        buf410 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf411 = reinterpret_tensor(buf406, (1, 512, 768), (393216, 768, 1), 0); del buf406  # reuse
        buf413 = buf394; del buf394  # reuse
        buf414 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf415 = reinterpret_tensor(buf414, (1, 512, 1), (512, 1, 1), 0); del buf414  # reuse
        buf416 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_87, add_88, attention_output_20, hidden_states_156, hidden_states_162, hidden_states_164, mean_66, mul_34, mul_35, pow_23, qp_11, query_states, query_states_11, sqrt_33, sub_44, variance_22], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf411, buf415, buf407, primals_153, primals_65, buf400, buf402, primals_66, primals_67, primals_68, buf410, buf413, buf416, 512, 768, grid=grid(512), stream=stream0)
        del primals_153
        del primals_66
        buf417 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf416, reinterpret_tensor(primals_154, (768, 2304), (1, 768), 0), out=buf417)
        buf418 = reinterpret_tensor(buf411, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf411  # reuse
        buf471 = reinterpret_tensor(buf407, (12, 64, 512), (64, 1, 768), 0); del buf407  # reuse
        # Source Nodes: [query_layer_34, query_layer_35, scale], Original ATen: [aten.add, aten.div, aten.sqrt, aten.transpose]
        triton_poi_fused_add_div_sqrt_transpose_2.run(buf417, primals_69, buf418, buf471, 393216, grid=grid(393216), stream=stream0)
        del primals_69
        buf419 = reinterpret_tensor(buf385, (12, 512, 512), (262144, 512, 1), 0); del buf385  # reuse
        # Source Nodes: [attention_scores_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf418, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf417, (12, 64, 512), (192, 1, 2304), 64), out=buf419)
        aten.bernoulli_(buf422, 0.9)
        buf425 = empty((1, 12, 512, 512), device='cuda', dtype=torch.bool)
        buf426 = reinterpret_tensor(buf382, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf382  # reuse
        buf470 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_22, attention_probs_23, query_states], Original ATen: [aten._softmax, aten._to_copy, aten.bitwise_not, aten.detach, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
        triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4.run(buf419, buf422, buf425, buf426, buf470, 6144, 512, grid=grid(6144), stream=stream0)
        del buf419
        del buf422
        buf427 = buf418; del buf418  # reuse
        buf469 = empty_strided((12, 64, 512), (64, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_layer_23], Original ATen: [aten.add, aten.transpose]
        triton_poi_fused_add_transpose_5.run(buf417, primals_70, buf427, buf469, 393216, grid=grid(393216), stream=stream0)
        del primals_70
        buf428 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf426, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf427, (12, 512, 64), (32768, 64, 1), 0), out=buf428)
        buf429 = reinterpret_tensor(buf427, (512, 768), (768, 1), 0); del buf427  # reuse
        # Source Nodes: [hidden_states_168], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf428, buf429, 393216, grid=grid(393216), stream=stream0)
        buf430 = reinterpret_tensor(buf428, (512, 768), (768, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf429, reinterpret_tensor(primals_155, (768, 768), (1, 768), 0), out=buf430)
        buf431 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf444 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_169, hidden_states_177], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_11.run(buf4, buf431, buf444, 393216, grid=grid(393216), stream=stream0)
        aten.bernoulli_(buf431, 0.9)
        buf434 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf435 = reinterpret_tensor(buf430, (1, 512, 768), (393216, 768, 1), 0); del buf430  # reuse
        buf437 = buf4; del buf4  # reuse
        buf438 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf439 = reinterpret_tensor(buf438, (1, 512, 1), (512, 1, 1), 0); del buf438  # reuse
        buf440 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_92, add_93, attention_output_22, hidden_states_164, hidden_states_169, hidden_states_171, hidden_states_174, mean_69, mul_35, mul_37, pow_24, query_states, query_states_11, sqrt_35, sub_46, variance_23], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf435, buf439, buf431, primals_156, primals_67, buf413, buf415, primals_68, primals_71, primals_72, buf434, buf437, buf440, 512, 768, grid=grid(512), stream=stream0)
        del primals_156
        del primals_68
        buf441 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_174], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf440, reinterpret_tensor(primals_157, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf441)
        del primals_158
        buf442 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_176, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf441, buf442, 1572864, grid=grid(1572864), stream=stream0)
        buf443 = reinterpret_tensor(buf435, (512, 768), (768, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf442, reinterpret_tensor(primals_159, (3072, 768), (1, 3072), 0), out=buf443)
        aten.bernoulli_(buf444, 0.9)
        buf447 = empty((1, 512, 768), device='cuda', dtype=torch.bool)
        buf448 = reinterpret_tensor(buf443, (1, 512, 768), (393216, 768, 1), 0); del buf443  # reuse
        buf450 = buf431; del buf431  # reuse
        buf451 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf452 = reinterpret_tensor(buf451, (1, 512, 1), (512, 1, 1), 0); del buf451  # reuse
        buf453 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_95, add_96, attention_output_22, hidden_states_171, hidden_states_177, hidden_states_179, logits, mean_72, mul_37, mul_38, pow_25, query_states, sequence_output, sqrt_36, sub_48, variance_24], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
        triton_per_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_9.run(buf448, buf452, buf444, primals_160, primals_71, buf437, buf439, primals_72, primals_73, primals_74, buf447, buf450, buf453, 512, 768, grid=grid(512), stream=stream0)
        del buf444
        del buf448
        del primals_160
        del primals_72
        del primals_74
        buf454 = empty((512, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf453, reinterpret_tensor(primals_161, (768, 2), (1, 768), 0), out=buf454)
        buf455 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf459 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_12.run(buf454, primals_162, buf455, buf459, 1, 512, grid=grid(1), stream=stream0)
        buf456 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf463 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_13.run(buf454, primals_162, buf456, buf463, 1, 512, grid=grid(1), stream=stream0)
        del buf454
        del primals_162
        buf460 = empty((1, ), device='cuda', dtype=torch.bool)
        buf464 = empty((1, ), device='cuda', dtype=torch.bool)
        buf505 = empty((), device='cuda', dtype=torch.float32)
        buf465 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf466 = empty((1, 1), device='cuda', dtype=torch.int64)
        buf467 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf468 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [add_98, end_loss, end_positions, loss, query_states, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_14.run(primals_165, primals_166, buf459, buf463, buf460, buf464, buf505, buf465, buf466, buf467, buf468, 1, grid=grid(1), stream=stream0)
        del primals_165
        del primals_166
        return (buf505, buf455, buf456, primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_164, primals_163, buf1, buf3, buf8, reinterpret_tensor(buf9, (512, 768), (768, 1), 0), buf19, buf23, buf28, buf30, buf32, buf33, buf34, buf35, buf40, buf43, buf45, buf46, buf55, buf59, buf64, buf67, buf69, buf70, buf71, buf72, buf77, buf80, buf82, buf83, buf92, buf96, buf101, buf104, buf106, buf107, buf108, buf109, buf114, buf117, buf119, buf120, buf129, buf133, buf138, buf141, buf143, buf144, buf145, buf146, buf151, buf154, buf156, buf157, buf166, buf170, buf175, buf178, buf180, buf181, buf182, buf183, buf188, buf191, buf193, buf194, buf203, buf207, buf212, buf215, buf217, buf218, buf219, buf220, buf225, buf228, buf230, buf231, buf240, buf244, buf249, buf252, buf254, buf255, buf256, buf257, buf262, buf265, buf267, buf268, buf277, buf281, buf286, buf289, buf291, buf292, buf293, buf294, buf299, buf302, buf304, buf305, buf314, buf318, buf323, buf326, buf328, buf329, buf330, buf331, buf336, buf339, buf341, buf342, buf351, buf355, buf360, buf363, buf365, buf366, buf367, buf368, buf373, buf376, buf378, buf379, buf388, buf392, buf397, buf400, buf402, buf403, buf404, buf405, buf410, buf413, buf415, buf416, buf425, buf429, buf434, buf437, buf439, buf440, buf441, buf442, buf447, buf450, buf452, buf453, buf459, buf460, buf463, buf464, buf465, buf466, buf467, buf468, reinterpret_tensor(primals_161, (2, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_157, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_155, (768, 768), (768, 1), 0), reinterpret_tensor(buf426, (12, 512, 512), (262144, 1, 512), 0), buf469, buf470, buf471, reinterpret_tensor(buf417, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_154, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_152, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_150, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_148, (768, 768), (768, 1), 0), reinterpret_tensor(buf389, (12, 512, 512), (262144, 1, 512), 0), buf472, buf473, buf474, reinterpret_tensor(buf380, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_147, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_141, (768, 768), (768, 1), 0), reinterpret_tensor(buf352, (12, 512, 512), (262144, 1, 512), 0), buf475, buf476, buf477, reinterpret_tensor(buf343, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_140, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_138, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_136, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_134, (768, 768), (768, 1), 0), reinterpret_tensor(buf315, (12, 512, 512), (262144, 1, 512), 0), buf478, buf479, buf480, reinterpret_tensor(buf306, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_133, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_131, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_129, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_127, (768, 768), (768, 1), 0), reinterpret_tensor(buf278, (12, 512, 512), (262144, 1, 512), 0), buf481, buf482, buf483, reinterpret_tensor(buf269, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_126, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_124, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 768), (768, 1), 0), reinterpret_tensor(buf241, (12, 512, 512), (262144, 1, 512), 0), buf484, buf485, buf486, reinterpret_tensor(buf232, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_119, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_117, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_115, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 768), (768, 1), 0), reinterpret_tensor(buf204, (12, 512, 512), (262144, 1, 512), 0), buf487, buf488, buf489, reinterpret_tensor(buf195, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_112, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_110, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_108, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), reinterpret_tensor(buf167, (12, 512, 512), (262144, 1, 512), 0), buf490, buf491, buf492, reinterpret_tensor(buf158, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_105, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_103, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_101, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(buf130, (12, 512, 512), (262144, 1, 512), 0), buf493, buf494, buf495, reinterpret_tensor(buf121, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_98, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_96, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_94, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(buf93, (12, 512, 512), (262144, 1, 512), 0), buf496, buf497, buf498, reinterpret_tensor(buf84, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_91, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_87, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 768), (768, 1), 0), reinterpret_tensor(buf56, (12, 512, 512), (262144, 1, 512), 0), buf499, buf500, buf501, reinterpret_tensor(buf47, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_84, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_82, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_78, (768, 768), (768, 1), 0), reinterpret_tensor(buf20, (12, 512, 512), (262144, 1, 512), 0), buf502, buf503, buf504, reinterpret_tensor(buf10, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_77, (2304, 768), (768, 1), 0), )


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
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_164 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_165 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_166 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForQuestionAnswering', benchmark_compiled_module)
