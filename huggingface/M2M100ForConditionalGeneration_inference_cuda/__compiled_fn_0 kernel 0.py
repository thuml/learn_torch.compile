
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lxxcktaofbyxibvcliuk3lkog5pj2nxsyn5vc7qawagfsam3pn.py
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
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyvorinf77dv2opbx6fvwbvvztmqtzcmpf3dwomqeigoh4eawci.py
# Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# hidden_states => add_2
# hidden_states_2 => add_3, add_4, mul_2, mul_3, rsqrt, sub, var_mean
# inputs_embeds => mul
# l__mod___model_encoder_embed_tokens => embedding
triton_red_fused_add_embedding_mul_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp23_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 128112
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 128112")
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 32.0
        tmp6 = tmp4 * tmp5
        tmp8 = tmp7.to(tl.int32)
        tmp9 = tl.full([1, 1], 0, tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.full([1, 1], 1, tl.int64)
        tmp12 = tmp0 != tmp11
        tmp13 = tmp12.to(tl.int32)
        tmp14 = tmp10 * tmp13
        tmp15 = tmp14.to(tl.int64)
        tmp16 = tmp15 + tmp11
        tmp17 = tmp16 + 1026
        tmp18 = tmp16 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp16)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp19 < 1026")
        tmp20 = tl.load(in_ptr3 + (r1 + (1024*tmp19)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp6 + tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp23_mean_next, tmp23_m2_next, tmp23_weight_next = triton_helpers.welford_reduce(
            tmp22, tmp23_mean, tmp23_m2, tmp23_weight,
        )
        tmp23_mean = tl.where(rmask & xmask, tmp23_mean_next, tmp23_mean)
        tmp23_m2 = tl.where(rmask & xmask, tmp23_m2_next, tmp23_m2)
        tmp23_weight = tl.where(rmask & xmask, tmp23_weight_next, tmp23_weight)
    tmp23_tmp, tmp24_tmp, tmp25_tmp = triton_helpers.welford(
        tmp23_mean, tmp23_m2, tmp23_weight, 1
    )
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp53 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp0 + 128112
        tmp27 = tmp0 < 0
        tmp28 = tl.where(tmp27, tmp26, tmp0)
        tl.device_assert(((0 <= tmp28) & (tmp28 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp28 < 128112")
        tmp29 = tl.load(in_ptr1 + (r1 + (1024*tmp28)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = 32.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp7.to(tl.int32)
        tmp33 = tl.full([1, 1], 0, tl.int32)
        tmp34 = tmp32 + tmp33
        tmp35 = tl.full([1, 1], 1, tl.int64)
        tmp36 = tmp0 != tmp35
        tmp37 = tmp36.to(tl.int32)
        tmp38 = tmp34 * tmp37
        tmp39 = tmp38.to(tl.int64)
        tmp40 = tmp39 + tmp35
        tmp41 = tmp40 + 1026
        tmp42 = tmp40 < 0
        tmp43 = tl.where(tmp42, tmp41, tmp40)
        tl.device_assert(((0 <= tmp43) & (tmp43 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp43 < 1026")
        tmp44 = tl.load(in_ptr3 + (r1 + (1024*tmp43)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp45 = tmp31 + tmp44
        tmp46 = tmp45 - tmp23
        tmp47 = 1024.0
        tmp48 = tmp24 / tmp47
        tmp49 = 1e-05
        tmp50 = tmp48 + tmp49
        tmp51 = tl.math.rsqrt(tmp50)
        tmp52 = tmp46 * tmp51
        tmp54 = tmp52 * tmp53
        tmp56 = tmp54 + tmp55
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp56, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqdoleigcqkgflnv67ktrqkstardltywhgf42tdpq2sxbqojvlh.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vwm33723czrwazqfqts67jk2gwvjrt2ypfiikwwibm7rxojeb6.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crndbflivgje6r7gdevjx4kpbxyjq2awam2csilb2azj2ex7yuud.py
# Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# attn_output_3 => clone_5
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlasip3mne4suhgzmgwushojouoiynbtwhzsgi3rxi2lbw7irf7.py
# Source Nodes: [hidden_states, hidden_states_6, inputs_embeds, l__mod___model_encoder_embed_tokens, residual_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# hidden_states => add_2
# hidden_states_6 => add_6, add_7, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
# inputs_embeds => mul
# l__mod___model_encoder_embed_tokens => embedding
# residual_1 => add_5
triton_per_fused_add_embedding_mul_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mul_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 128112
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 128112")
    tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, other=0.0)
    tmp5 = 32.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp0 != tmp11
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15 + tmp11
    tmp17 = tmp16 + 1026
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp19 < 1026")
    tmp20 = tl.load(in_ptr3 + (r1 + (1024*tmp19)), rmask & xmask, other=0.0)
    tmp21 = tmp6 + tmp20
    tmp24 = tmp22 + tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tl.full([1], 1024, tl.int32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 / tmp34
    tmp36 = tmp26 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp42 = tmp25 - tmp35
    tmp43 = 1024.0
    tmp44 = tmp41 / tmp43
    tmp45 = 1e-05
    tmp46 = tmp44 + tmp45
    tmp47 = tl.math.rsqrt(tmp46)
    tmp48 = tmp42 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp52, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlbqkm3yj5quwjq4ajn5g3wrkkjjfzkwntnqpfsfavkeqpl4xh5.py
# Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
# hidden_states_7 => relu
triton_poi_fused_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvyak5lvhztyi6qxzi6c4h2rnt2wgyfiiq3fgknnycajyaj2ayv.py
# Source Nodes: [hidden_states_13, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_13 => add_10, add_9, mul_7, mul_8, rsqrt_2, sub_3, var_mean_2
# residual_2 => add_8
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpxgo7fq4hpn2mpakkzwubpehizqhy4wzucbu5wmst6fqgw4xzl.py
# Source Nodes: [hidden_states_17, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_17 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
# residual_2 => add_8
# residual_3 => add_11
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5o/c5oe2odncnn3ud6hos6ripjxaq3cwycw66qzokphrkp4harkubzg.py
# Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
# attn_weights_27 => amax_12, div_12, exp_12, sub_38, sum_13
triton_per_fused__softmax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask, other=0.0)
    tmp1 = r2
    tmp2 = 1 + x0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.0
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp7 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp13 / tmp17
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4xpkezncwum2ih565gkh4kwazncus7t47zhhycbmqpkv5jb4kx.py
# Source Nodes: [attn_output_63], Original ATen: [aten.clone]
# attn_output_63 => clone_102
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7d2bcxzjojuq4ozxwsktxv75ih66rslediwmf3unf6wdx4il5n.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_36
triton_red_fused__log_softmax_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32028
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32028*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwevmpbiv3eh2r6rm4jp3up7qyvy5blnjr3inwyteyyq22nqyope.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_36
triton_per_fused__log_softmax_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46xu654ke6p5z3q5gvzqsyokounvhpwj7k766f6t4lx55bwvrf6.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => exp_36, sub_98, sum_37
triton_red_fused__log_softmax_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32028
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 4)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (32028*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl.exp(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/66/c66epxixb7bwhidimaq3r5l6fkoavx63rjn3wmf2fmpdhapelgjw.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => exp_36, sub_98, sum_37
triton_per_fused__log_softmax_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/53/c53eo5sgjj3vdax7pszx5w7p5v4qbazfimhv6b2dmg7gymnum4vh.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type_6, div_36, full_default_3, ne_3, ne_4, neg, sum_38, sum_39, where_2
triton_per_fused_nll_loss_forward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 128112
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 128112), "index out of bounds: 0 <= tmp7 < 128112")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (128112*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.log(tmp11)
    tmp13 = tmp10 - tmp12
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp2.to(tl.int64)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128112, 1024), (1024, 1))
    assert_size_stride(arg1_1, (1024, ), (1, ))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg3_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg14_1, (4096, ), (1, ))
    assert_size_stride(arg15_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg16_1, (1024, ), (1, ))
    assert_size_stride(arg17_1, (1024, ), (1, ))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg30_1, (4096, ), (1, ))
    assert_size_stride(arg31_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg32_1, (1024, ), (1, ))
    assert_size_stride(arg33_1, (1024, ), (1, ))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, ), (1, ))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg46_1, (4096, ), (1, ))
    assert_size_stride(arg47_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg62_1, (4096, ), (1, ))
    assert_size_stride(arg63_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg78_1, (4096, ), (1, ))
    assert_size_stride(arg79_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, ), (1, ))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg94_1, (4096, ), (1, ))
    assert_size_stride(arg95_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg96_1, (1024, ), (1, ))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg110_1, (4096, ), (1, ))
    assert_size_stride(arg111_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg126_1, (4096, ), (1, ))
    assert_size_stride(arg127_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (1024, ), (1, ))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg142_1, (4096, ), (1, ))
    assert_size_stride(arg143_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg158_1, (4096, ), (1, ))
    assert_size_stride(arg159_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg174_1, (4096, ), (1, ))
    assert_size_stride(arg175_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg190_1, (4096, ), (1, ))
    assert_size_stride(arg191_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (128112, 1024), (1024, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg203_1, (1024, ), (1, ))
    assert_size_stride(arg204_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg219_1, (4096, ), (1, ))
    assert_size_stride(arg220_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg221_1, (1024, ), (1, ))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg225_1, (1024, ), (1, ))
    assert_size_stride(arg226_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg229_1, (1024, ), (1, ))
    assert_size_stride(arg230_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg231_1, (1024, ), (1, ))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, ), (1, ))
    assert_size_stride(arg234_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg245_1, (4096, ), (1, ))
    assert_size_stride(arg246_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg247_1, (1024, ), (1, ))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg251_1, (1024, ), (1, ))
    assert_size_stride(arg252_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg271_1, (4096, ), (1, ))
    assert_size_stride(arg272_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg273_1, (1024, ), (1, ))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg277_1, (1024, ), (1, ))
    assert_size_stride(arg278_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg283_1, (1024, ), (1, ))
    assert_size_stride(arg284_1, (1024, ), (1, ))
    assert_size_stride(arg285_1, (1024, ), (1, ))
    assert_size_stride(arg286_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg291_1, (1024, ), (1, ))
    assert_size_stride(arg292_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg297_1, (4096, ), (1, ))
    assert_size_stride(arg298_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg299_1, (1024, ), (1, ))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg309_1, (1024, ), (1, ))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg315_1, (1024, ), (1, ))
    assert_size_stride(arg316_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg323_1, (4096, ), (1, ))
    assert_size_stride(arg324_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg329_1, (1024, ), (1, ))
    assert_size_stride(arg330_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg349_1, (4096, ), (1, ))
    assert_size_stride(arg350_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg357_1, (1024, ), (1, ))
    assert_size_stride(arg358_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg367_1, (1024, ), (1, ))
    assert_size_stride(arg368_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, ), (1, ))
    assert_size_stride(arg374_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg375_1, (4096, ), (1, ))
    assert_size_stride(arg376_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg381_1, (1024, ), (1, ))
    assert_size_stride(arg382_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg383_1, (1024, ), (1, ))
    assert_size_stride(arg384_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg385_1, (1024, ), (1, ))
    assert_size_stride(arg386_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (1024, ), (1, ))
    assert_size_stride(arg389_1, (1024, ), (1, ))
    assert_size_stride(arg390_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg397_1, (1024, ), (1, ))
    assert_size_stride(arg398_1, (1024, ), (1, ))
    assert_size_stride(arg399_1, (1024, ), (1, ))
    assert_size_stride(arg400_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg401_1, (4096, ), (1, ))
    assert_size_stride(arg402_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg403_1, (1024, ), (1, ))
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg407_1, (1024, ), (1, ))
    assert_size_stride(arg408_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg409_1, (1024, ), (1, ))
    assert_size_stride(arg410_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg411_1, (1024, ), (1, ))
    assert_size_stride(arg412_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg413_1, (1024, ), (1, ))
    assert_size_stride(arg414_1, (1024, ), (1, ))
    assert_size_stride(arg415_1, (1024, ), (1, ))
    assert_size_stride(arg416_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg419_1, (1024, ), (1, ))
    assert_size_stride(arg420_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (1024, ), (1, ))
    assert_size_stride(arg425_1, (1024, ), (1, ))
    assert_size_stride(arg426_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg427_1, (4096, ), (1, ))
    assert_size_stride(arg428_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg429_1, (1024, ), (1, ))
    assert_size_stride(arg430_1, (1024, ), (1, ))
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg433_1, (1024, ), (1, ))
    assert_size_stride(arg434_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg435_1, (1024, ), (1, ))
    assert_size_stride(arg436_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg439_1, (1024, ), (1, ))
    assert_size_stride(arg440_1, (1024, ), (1, ))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg443_1, (1024, ), (1, ))
    assert_size_stride(arg444_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg445_1, (1024, ), (1, ))
    assert_size_stride(arg446_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg447_1, (1024, ), (1, ))
    assert_size_stride(arg448_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg449_1, (1024, ), (1, ))
    assert_size_stride(arg450_1, (1024, ), (1, ))
    assert_size_stride(arg451_1, (1024, ), (1, ))
    assert_size_stride(arg452_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg453_1, (4096, ), (1, ))
    assert_size_stride(arg454_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg455_1, (1024, ), (1, ))
    assert_size_stride(arg456_1, (1024, ), (1, ))
    assert_size_stride(arg457_1, (1024, ), (1, ))
    assert_size_stride(arg458_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg459_1, (1024, ), (1, ))
    assert_size_stride(arg460_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg461_1, (1024, ), (1, ))
    assert_size_stride(arg462_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg463_1, (1024, ), (1, ))
    assert_size_stride(arg464_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg465_1, (1024, ), (1, ))
    assert_size_stride(arg466_1, (1024, ), (1, ))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg471_1, (1024, ), (1, ))
    assert_size_stride(arg472_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg473_1, (1024, ), (1, ))
    assert_size_stride(arg474_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg475_1, (1024, ), (1, ))
    assert_size_stride(arg476_1, (1024, ), (1, ))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg479_1, (4096, ), (1, ))
    assert_size_stride(arg480_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg481_1, (1024, ), (1, ))
    assert_size_stride(arg482_1, (1024, ), (1, ))
    assert_size_stride(arg483_1, (1024, ), (1, ))
    assert_size_stride(arg484_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg485_1, (1024, ), (1, ))
    assert_size_stride(arg486_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg487_1, (1024, ), (1, ))
    assert_size_stride(arg488_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg489_1, (1024, ), (1, ))
    assert_size_stride(arg490_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg491_1, (1024, ), (1, ))
    assert_size_stride(arg492_1, (1024, ), (1, ))
    assert_size_stride(arg493_1, (1024, ), (1, ))
    assert_size_stride(arg494_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg499_1, (1024, ), (1, ))
    assert_size_stride(arg500_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg501_1, (1024, ), (1, ))
    assert_size_stride(arg502_1, (1024, ), (1, ))
    assert_size_stride(arg503_1, (1024, ), (1, ))
    assert_size_stride(arg504_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg505_1, (4096, ), (1, ))
    assert_size_stride(arg506_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg507_1, (1024, ), (1, ))
    assert_size_stride(arg508_1, (1024, ), (1, ))
    assert_size_stride(arg509_1, (1024, ), (1, ))
    assert_size_stride(arg510_1, (128112, 1024), (1024, 1))
    assert_size_stride(arg511_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg512_1, (1026, 1024), (1024, 1))
    assert_size_stride(arg513_1, (1, 128), (128, 1))
    assert_size_stride(arg514_1, (1, 128), (128, 1))
    assert_size_stride(arg515_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128), device='cuda', dtype=torch.int32)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_cumsum_ne_0.run(arg515_1, buf0, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf1 = aten.cumsum(buf0, 1)
        buf2 = buf1
        del buf1
        buf6 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, l__mod___model_encoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_embedding_mul_native_layer_norm_1.run(arg515_1, arg0_1, buf2, arg511_1, arg1_1, arg2_1, buf6, 128, 1024, grid=grid(128), stream=stream0)
        del arg1_1
        del arg2_1
        buf7 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg3_1, (1024, 1024), (1, 1024), 0), out=buf7)
        del arg3_1
        buf8 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg5_1, (1024, 1024), (1, 1024), 0), out=buf8)
        del arg5_1
        buf9 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), out=buf9)
        del arg7_1
        buf10 = reinterpret_tensor(buf6, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf7, arg4_1, buf10, 131072, grid=grid(131072), stream=stream0)
        del arg4_1
        buf11 = reinterpret_tensor(buf7, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf8, arg6_1, buf11, 131072, grid=grid(131072), stream=stream0)
        del arg6_1
        buf12 = reinterpret_tensor(buf8, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf9, arg8_1, buf12, 131072, grid=grid(131072), stream=stream0)
        del arg8_1
        # Source Nodes: [], Original ATen: []
        buf13 = aten._scaled_dot_product_efficient_attention(buf10, buf11, buf12, None, True, scale=1.0)
        buf14 = buf13[0]
        del buf13
        buf18 = reinterpret_tensor(buf14, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf18, 131072, grid=grid(131072), stream=stream0)
        buf19 = reinterpret_tensor(buf12, (128, 1024), (1024, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), out=buf19)
        del arg9_1
        buf20 = reinterpret_tensor(buf19, (1, 128, 1024), (131072, 1024, 1), 0); del buf19  # reuse
        buf24 = reinterpret_tensor(buf18, (1, 128, 1024), (131072, 1024, 1), 0); del buf18  # reuse
        # Source Nodes: [hidden_states, hidden_states_6, inputs_embeds, l__mod___model_encoder_embed_tokens, residual_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_embedding_mul_native_layer_norm_5.run(buf20, arg515_1, arg0_1, buf2, arg511_1, arg10_1, arg11_1, arg12_1, buf24, 128, 1024, grid=grid(128), stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg12_1
        del arg511_1
        del arg515_1
        del buf2
        buf25 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 4096), (1, 1024), 0), out=buf25)
        del arg13_1
        buf26 = reinterpret_tensor(buf25, (1, 128, 4096), (524288, 4096, 1), 0); del buf25  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf26, arg14_1, 524288, grid=grid(524288), stream=stream0)
        del arg14_1
        buf27 = reinterpret_tensor(buf24, (128, 1024), (1024, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg15_1, (4096, 1024), (1, 4096), 0), out=buf27)
        del arg15_1
        buf31 = reinterpret_tensor(buf11, (1, 128, 1024), (131072, 1024, 1), 0); del buf11  # reuse
        # Source Nodes: [hidden_states_13, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf20, buf27, arg16_1, arg17_1, arg18_1, buf31, 128, 1024, grid=grid(128), stream=stream0)
        del arg17_1
        del arg18_1
        buf32 = reinterpret_tensor(buf10, (128, 1024), (1024, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg19_1, (1024, 1024), (1, 1024), 0), out=buf32)
        del arg19_1
        buf33 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg21_1, (1024, 1024), (1, 1024), 0), out=buf33)
        del arg21_1
        buf34 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), out=buf34)
        del arg23_1
        buf35 = reinterpret_tensor(buf31, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf32, arg20_1, buf35, 131072, grid=grid(131072), stream=stream0)
        del arg20_1
        buf36 = reinterpret_tensor(buf32, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf33, arg22_1, buf36, 131072, grid=grid(131072), stream=stream0)
        del arg22_1
        buf37 = reinterpret_tensor(buf33, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf34, arg24_1, buf37, 131072, grid=grid(131072), stream=stream0)
        del arg24_1
        # Source Nodes: [], Original ATen: []
        buf38 = aten._scaled_dot_product_efficient_attention(buf35, buf36, buf37, None, True, scale=1.0)
        buf39 = buf38[0]
        del buf38
        buf43 = reinterpret_tensor(buf39, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf39  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf43, 131072, grid=grid(131072), stream=stream0)
        buf44 = reinterpret_tensor(buf37, (128, 1024), (1024, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf43, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), out=buf44)
        del arg25_1
        buf45 = reinterpret_tensor(buf44, (1, 128, 1024), (131072, 1024, 1), 0); del buf44  # reuse
        buf49 = reinterpret_tensor(buf43, (1, 128, 1024), (131072, 1024, 1), 0); del buf43  # reuse
        # Source Nodes: [hidden_states_17, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf45, buf20, buf27, arg16_1, arg26_1, arg27_1, arg28_1, buf49, 128, 1024, grid=grid(128), stream=stream0)
        del arg16_1
        del arg26_1
        del arg27_1
        del arg28_1
        buf50 = reinterpret_tensor(buf26, (128, 4096), (4096, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg29_1, (1024, 4096), (1, 1024), 0), out=buf50)
        del arg29_1
        buf51 = reinterpret_tensor(buf50, (1, 128, 4096), (524288, 4096, 1), 0); del buf50  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf51, arg30_1, 524288, grid=grid(524288), stream=stream0)
        del arg30_1
        buf52 = reinterpret_tensor(buf49, (128, 1024), (1024, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg31_1, (4096, 1024), (1, 4096), 0), out=buf52)
        del arg31_1
        buf56 = reinterpret_tensor(buf27, (1, 128, 1024), (131072, 1024, 1), 0); del buf27  # reuse
        # Source Nodes: [hidden_states_24, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf45, buf52, arg32_1, arg33_1, arg34_1, buf56, 128, 1024, grid=grid(128), stream=stream0)
        del arg33_1
        del arg34_1
        buf57 = reinterpret_tensor(buf20, (128, 1024), (1024, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg35_1, (1024, 1024), (1, 1024), 0), out=buf57)
        del arg35_1
        buf58 = reinterpret_tensor(buf36, (128, 1024), (1024, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 1024), (1, 1024), 0), out=buf58)
        del arg37_1
        buf59 = reinterpret_tensor(buf35, (128, 1024), (1024, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), out=buf59)
        del arg39_1
        buf60 = reinterpret_tensor(buf56, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf57, arg36_1, buf60, 131072, grid=grid(131072), stream=stream0)
        del arg36_1
        buf61 = reinterpret_tensor(buf57, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf58, arg38_1, buf61, 131072, grid=grid(131072), stream=stream0)
        del arg38_1
        buf62 = reinterpret_tensor(buf58, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf59, arg40_1, buf62, 131072, grid=grid(131072), stream=stream0)
        del arg40_1
        # Source Nodes: [], Original ATen: []
        buf63 = aten._scaled_dot_product_efficient_attention(buf60, buf61, buf62, None, True, scale=1.0)
        buf64 = buf63[0]
        del buf63
        buf68 = reinterpret_tensor(buf64, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf68, 131072, grid=grid(131072), stream=stream0)
        buf69 = reinterpret_tensor(buf62, (128, 1024), (1024, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), out=buf69)
        del arg41_1
        buf70 = reinterpret_tensor(buf69, (1, 128, 1024), (131072, 1024, 1), 0); del buf69  # reuse
        buf74 = reinterpret_tensor(buf68, (1, 128, 1024), (131072, 1024, 1), 0); del buf68  # reuse
        # Source Nodes: [hidden_states_28, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf70, buf45, buf52, arg32_1, arg42_1, arg43_1, arg44_1, buf74, 128, 1024, grid=grid(128), stream=stream0)
        del arg32_1
        del arg42_1
        del arg43_1
        del arg44_1
        buf75 = reinterpret_tensor(buf51, (128, 4096), (4096, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg45_1, (1024, 4096), (1, 1024), 0), out=buf75)
        del arg45_1
        buf76 = reinterpret_tensor(buf75, (1, 128, 4096), (524288, 4096, 1), 0); del buf75  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf76, arg46_1, 524288, grid=grid(524288), stream=stream0)
        del arg46_1
        buf77 = reinterpret_tensor(buf74, (128, 1024), (1024, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg47_1, (4096, 1024), (1, 4096), 0), out=buf77)
        del arg47_1
        buf81 = reinterpret_tensor(buf52, (1, 128, 1024), (131072, 1024, 1), 0); del buf52  # reuse
        # Source Nodes: [hidden_states_35, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf70, buf77, arg48_1, arg49_1, arg50_1, buf81, 128, 1024, grid=grid(128), stream=stream0)
        del arg49_1
        del arg50_1
        buf82 = reinterpret_tensor(buf45, (128, 1024), (1024, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg51_1, (1024, 1024), (1, 1024), 0), out=buf82)
        del arg51_1
        buf83 = reinterpret_tensor(buf61, (128, 1024), (1024, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg53_1, (1024, 1024), (1, 1024), 0), out=buf83)
        del arg53_1
        buf84 = reinterpret_tensor(buf60, (128, 1024), (1024, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), out=buf84)
        del arg55_1
        buf85 = reinterpret_tensor(buf81, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf82, arg52_1, buf85, 131072, grid=grid(131072), stream=stream0)
        del arg52_1
        buf86 = reinterpret_tensor(buf82, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf83, arg54_1, buf86, 131072, grid=grid(131072), stream=stream0)
        del arg54_1
        buf87 = reinterpret_tensor(buf83, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf84, arg56_1, buf87, 131072, grid=grid(131072), stream=stream0)
        del arg56_1
        # Source Nodes: [], Original ATen: []
        buf88 = aten._scaled_dot_product_efficient_attention(buf85, buf86, buf87, None, True, scale=1.0)
        buf89 = buf88[0]
        del buf88
        buf93 = reinterpret_tensor(buf89, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf89  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf93, 131072, grid=grid(131072), stream=stream0)
        buf94 = reinterpret_tensor(buf87, (128, 1024), (1024, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), out=buf94)
        del arg57_1
        buf95 = reinterpret_tensor(buf94, (1, 128, 1024), (131072, 1024, 1), 0); del buf94  # reuse
        buf99 = reinterpret_tensor(buf93, (1, 128, 1024), (131072, 1024, 1), 0); del buf93  # reuse
        # Source Nodes: [hidden_states_39, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf95, buf70, buf77, arg48_1, arg58_1, arg59_1, arg60_1, buf99, 128, 1024, grid=grid(128), stream=stream0)
        del arg48_1
        del arg58_1
        del arg59_1
        del arg60_1
        buf100 = reinterpret_tensor(buf76, (128, 4096), (4096, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 4096), (1, 1024), 0), out=buf100)
        del arg61_1
        buf101 = reinterpret_tensor(buf100, (1, 128, 4096), (524288, 4096, 1), 0); del buf100  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf101, arg62_1, 524288, grid=grid(524288), stream=stream0)
        del arg62_1
        buf102 = reinterpret_tensor(buf99, (128, 1024), (1024, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg63_1, (4096, 1024), (1, 4096), 0), out=buf102)
        del arg63_1
        buf106 = reinterpret_tensor(buf77, (1, 128, 1024), (131072, 1024, 1), 0); del buf77  # reuse
        # Source Nodes: [hidden_states_46, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf95, buf102, arg64_1, arg65_1, arg66_1, buf106, 128, 1024, grid=grid(128), stream=stream0)
        del arg65_1
        del arg66_1
        buf107 = reinterpret_tensor(buf70, (128, 1024), (1024, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg67_1, (1024, 1024), (1, 1024), 0), out=buf107)
        del arg67_1
        buf108 = reinterpret_tensor(buf86, (128, 1024), (1024, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 1024), (1, 1024), 0), out=buf108)
        del arg69_1
        buf109 = reinterpret_tensor(buf85, (128, 1024), (1024, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), out=buf109)
        del arg71_1
        buf110 = reinterpret_tensor(buf106, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf107, arg68_1, buf110, 131072, grid=grid(131072), stream=stream0)
        del arg68_1
        buf111 = reinterpret_tensor(buf107, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf108, arg70_1, buf111, 131072, grid=grid(131072), stream=stream0)
        del arg70_1
        buf112 = reinterpret_tensor(buf108, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf109, arg72_1, buf112, 131072, grid=grid(131072), stream=stream0)
        del arg72_1
        # Source Nodes: [], Original ATen: []
        buf113 = aten._scaled_dot_product_efficient_attention(buf110, buf111, buf112, None, True, scale=1.0)
        buf114 = buf113[0]
        del buf113
        buf118 = reinterpret_tensor(buf114, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf114  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf118, 131072, grid=grid(131072), stream=stream0)
        buf119 = reinterpret_tensor(buf112, (128, 1024), (1024, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), out=buf119)
        del arg73_1
        buf120 = reinterpret_tensor(buf119, (1, 128, 1024), (131072, 1024, 1), 0); del buf119  # reuse
        buf124 = reinterpret_tensor(buf118, (1, 128, 1024), (131072, 1024, 1), 0); del buf118  # reuse
        # Source Nodes: [hidden_states_50, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf120, buf95, buf102, arg64_1, arg74_1, arg75_1, arg76_1, buf124, 128, 1024, grid=grid(128), stream=stream0)
        del arg64_1
        del arg74_1
        del arg75_1
        del arg76_1
        buf125 = reinterpret_tensor(buf101, (128, 4096), (4096, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 4096), (1, 1024), 0), out=buf125)
        del arg77_1
        buf126 = reinterpret_tensor(buf125, (1, 128, 4096), (524288, 4096, 1), 0); del buf125  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf126, arg78_1, 524288, grid=grid(524288), stream=stream0)
        del arg78_1
        buf127 = reinterpret_tensor(buf124, (128, 1024), (1024, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg79_1, (4096, 1024), (1, 4096), 0), out=buf127)
        del arg79_1
        buf131 = buf95; del buf95  # reuse
        # Source Nodes: [hidden_states_57, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf120, buf127, arg80_1, arg81_1, arg82_1, buf131, 128, 1024, grid=grid(128), stream=stream0)
        del arg81_1
        del arg82_1
        buf132 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg83_1, (1024, 1024), (1, 1024), 0), out=buf132)
        del arg83_1
        buf133 = reinterpret_tensor(buf111, (128, 1024), (1024, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 1024), (1, 1024), 0), out=buf133)
        del arg85_1
        buf134 = reinterpret_tensor(buf110, (128, 1024), (1024, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), out=buf134)
        del arg87_1
        buf135 = reinterpret_tensor(buf131, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf132, arg84_1, buf135, 131072, grid=grid(131072), stream=stream0)
        del arg84_1
        buf136 = reinterpret_tensor(buf132, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf133, arg86_1, buf136, 131072, grid=grid(131072), stream=stream0)
        del arg86_1
        buf137 = reinterpret_tensor(buf133, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf134, arg88_1, buf137, 131072, grid=grid(131072), stream=stream0)
        del arg88_1
        # Source Nodes: [], Original ATen: []
        buf138 = aten._scaled_dot_product_efficient_attention(buf135, buf136, buf137, None, True, scale=1.0)
        buf139 = buf138[0]
        del buf138
        buf143 = reinterpret_tensor(buf139, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf139  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf143, 131072, grid=grid(131072), stream=stream0)
        buf144 = reinterpret_tensor(buf137, (128, 1024), (1024, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), out=buf144)
        del arg89_1
        buf145 = reinterpret_tensor(buf144, (1, 128, 1024), (131072, 1024, 1), 0); del buf144  # reuse
        buf149 = reinterpret_tensor(buf143, (1, 128, 1024), (131072, 1024, 1), 0); del buf143  # reuse
        # Source Nodes: [hidden_states_61, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf145, buf120, buf127, arg80_1, arg90_1, arg91_1, arg92_1, buf149, 128, 1024, grid=grid(128), stream=stream0)
        del arg80_1
        del arg90_1
        del arg91_1
        del arg92_1
        buf150 = reinterpret_tensor(buf126, (128, 4096), (4096, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg93_1, (1024, 4096), (1, 1024), 0), out=buf150)
        del arg93_1
        buf151 = reinterpret_tensor(buf150, (1, 128, 4096), (524288, 4096, 1), 0); del buf150  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf151, arg94_1, 524288, grid=grid(524288), stream=stream0)
        del arg94_1
        buf152 = reinterpret_tensor(buf149, (128, 1024), (1024, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg95_1, (4096, 1024), (1, 4096), 0), out=buf152)
        del arg95_1
        buf156 = reinterpret_tensor(buf127, (1, 128, 1024), (131072, 1024, 1), 0); del buf127  # reuse
        # Source Nodes: [hidden_states_68, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf145, buf152, arg96_1, arg97_1, arg98_1, buf156, 128, 1024, grid=grid(128), stream=stream0)
        del arg97_1
        del arg98_1
        buf157 = reinterpret_tensor(buf120, (128, 1024), (1024, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg99_1, (1024, 1024), (1, 1024), 0), out=buf157)
        del arg99_1
        buf158 = reinterpret_tensor(buf136, (128, 1024), (1024, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 1024), (1, 1024), 0), out=buf158)
        del arg101_1
        buf159 = reinterpret_tensor(buf135, (128, 1024), (1024, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), out=buf159)
        del arg103_1
        buf160 = reinterpret_tensor(buf156, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf157, arg100_1, buf160, 131072, grid=grid(131072), stream=stream0)
        del arg100_1
        buf161 = reinterpret_tensor(buf157, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf158, arg102_1, buf161, 131072, grid=grid(131072), stream=stream0)
        del arg102_1
        buf162 = reinterpret_tensor(buf158, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf159, arg104_1, buf162, 131072, grid=grid(131072), stream=stream0)
        del arg104_1
        # Source Nodes: [], Original ATen: []
        buf163 = aten._scaled_dot_product_efficient_attention(buf160, buf161, buf162, None, True, scale=1.0)
        buf164 = buf163[0]
        del buf163
        buf168 = reinterpret_tensor(buf164, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf164  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf168, 131072, grid=grid(131072), stream=stream0)
        buf169 = reinterpret_tensor(buf162, (128, 1024), (1024, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), out=buf169)
        del arg105_1
        buf170 = reinterpret_tensor(buf169, (1, 128, 1024), (131072, 1024, 1), 0); del buf169  # reuse
        buf174 = reinterpret_tensor(buf168, (1, 128, 1024), (131072, 1024, 1), 0); del buf168  # reuse
        # Source Nodes: [hidden_states_72, residual_12, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf170, buf145, buf152, arg96_1, arg106_1, arg107_1, arg108_1, buf174, 128, 1024, grid=grid(128), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        del arg96_1
        buf175 = reinterpret_tensor(buf151, (128, 4096), (4096, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg109_1, (1024, 4096), (1, 1024), 0), out=buf175)
        del arg109_1
        buf176 = reinterpret_tensor(buf175, (1, 128, 4096), (524288, 4096, 1), 0); del buf175  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf176, arg110_1, 524288, grid=grid(524288), stream=stream0)
        del arg110_1
        buf177 = reinterpret_tensor(buf174, (128, 1024), (1024, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg111_1, (4096, 1024), (1, 4096), 0), out=buf177)
        del arg111_1
        buf181 = reinterpret_tensor(buf152, (1, 128, 1024), (131072, 1024, 1), 0); del buf152  # reuse
        # Source Nodes: [hidden_states_79, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf170, buf177, arg112_1, arg113_1, arg114_1, buf181, 128, 1024, grid=grid(128), stream=stream0)
        del arg113_1
        del arg114_1
        buf182 = reinterpret_tensor(buf145, (128, 1024), (1024, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg115_1, (1024, 1024), (1, 1024), 0), out=buf182)
        del arg115_1
        buf183 = reinterpret_tensor(buf161, (128, 1024), (1024, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 1024), (1, 1024), 0), out=buf183)
        del arg117_1
        buf184 = reinterpret_tensor(buf160, (128, 1024), (1024, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), out=buf184)
        del arg119_1
        buf185 = reinterpret_tensor(buf181, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf182, arg116_1, buf185, 131072, grid=grid(131072), stream=stream0)
        del arg116_1
        buf186 = reinterpret_tensor(buf182, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf183, arg118_1, buf186, 131072, grid=grid(131072), stream=stream0)
        del arg118_1
        buf187 = reinterpret_tensor(buf183, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf184, arg120_1, buf187, 131072, grid=grid(131072), stream=stream0)
        del arg120_1
        # Source Nodes: [], Original ATen: []
        buf188 = aten._scaled_dot_product_efficient_attention(buf185, buf186, buf187, None, True, scale=1.0)
        buf189 = buf188[0]
        del buf188
        buf193 = reinterpret_tensor(buf189, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf189  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf193, 131072, grid=grid(131072), stream=stream0)
        buf194 = reinterpret_tensor(buf187, (128, 1024), (1024, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), out=buf194)
        del arg121_1
        buf195 = reinterpret_tensor(buf194, (1, 128, 1024), (131072, 1024, 1), 0); del buf194  # reuse
        buf199 = reinterpret_tensor(buf193, (1, 128, 1024), (131072, 1024, 1), 0); del buf193  # reuse
        # Source Nodes: [hidden_states_83, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf195, buf170, buf177, arg112_1, arg122_1, arg123_1, arg124_1, buf199, 128, 1024, grid=grid(128), stream=stream0)
        del arg112_1
        del arg122_1
        del arg123_1
        del arg124_1
        buf200 = reinterpret_tensor(buf176, (128, 4096), (4096, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg125_1, (1024, 4096), (1, 1024), 0), out=buf200)
        del arg125_1
        buf201 = reinterpret_tensor(buf200, (1, 128, 4096), (524288, 4096, 1), 0); del buf200  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf201, arg126_1, 524288, grid=grid(524288), stream=stream0)
        del arg126_1
        buf202 = reinterpret_tensor(buf199, (128, 1024), (1024, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg127_1, (4096, 1024), (1, 4096), 0), out=buf202)
        del arg127_1
        buf206 = reinterpret_tensor(buf177, (1, 128, 1024), (131072, 1024, 1), 0); del buf177  # reuse
        # Source Nodes: [hidden_states_90, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf195, buf202, arg128_1, arg129_1, arg130_1, buf206, 128, 1024, grid=grid(128), stream=stream0)
        del arg129_1
        del arg130_1
        buf207 = reinterpret_tensor(buf170, (128, 1024), (1024, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg131_1, (1024, 1024), (1, 1024), 0), out=buf207)
        del arg131_1
        buf208 = reinterpret_tensor(buf186, (128, 1024), (1024, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 1024), (1, 1024), 0), out=buf208)
        del arg133_1
        buf209 = reinterpret_tensor(buf185, (128, 1024), (1024, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), out=buf209)
        del arg135_1
        buf210 = reinterpret_tensor(buf206, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf207, arg132_1, buf210, 131072, grid=grid(131072), stream=stream0)
        del arg132_1
        buf211 = reinterpret_tensor(buf207, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf208, arg134_1, buf211, 131072, grid=grid(131072), stream=stream0)
        del arg134_1
        buf212 = reinterpret_tensor(buf208, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf209, arg136_1, buf212, 131072, grid=grid(131072), stream=stream0)
        del arg136_1
        # Source Nodes: [], Original ATen: []
        buf213 = aten._scaled_dot_product_efficient_attention(buf210, buf211, buf212, None, True, scale=1.0)
        buf214 = buf213[0]
        del buf213
        buf218 = reinterpret_tensor(buf214, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf214  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf218, 131072, grid=grid(131072), stream=stream0)
        buf219 = reinterpret_tensor(buf212, (128, 1024), (1024, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf219)
        del arg137_1
        buf220 = reinterpret_tensor(buf219, (1, 128, 1024), (131072, 1024, 1), 0); del buf219  # reuse
        buf224 = reinterpret_tensor(buf218, (1, 128, 1024), (131072, 1024, 1), 0); del buf218  # reuse
        # Source Nodes: [hidden_states_94, residual_16, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf220, buf195, buf202, arg128_1, arg138_1, arg139_1, arg140_1, buf224, 128, 1024, grid=grid(128), stream=stream0)
        del arg128_1
        del arg138_1
        del arg139_1
        del arg140_1
        buf225 = reinterpret_tensor(buf201, (128, 4096), (4096, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 4096), (1, 1024), 0), out=buf225)
        del arg141_1
        buf226 = reinterpret_tensor(buf225, (1, 128, 4096), (524288, 4096, 1), 0); del buf225  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf226, arg142_1, 524288, grid=grid(524288), stream=stream0)
        del arg142_1
        buf227 = reinterpret_tensor(buf224, (128, 1024), (1024, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg143_1, (4096, 1024), (1, 4096), 0), out=buf227)
        del arg143_1
        buf231 = reinterpret_tensor(buf202, (1, 128, 1024), (131072, 1024, 1), 0); del buf202  # reuse
        # Source Nodes: [hidden_states_101, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf220, buf227, arg144_1, arg145_1, arg146_1, buf231, 128, 1024, grid=grid(128), stream=stream0)
        del arg145_1
        del arg146_1
        buf232 = reinterpret_tensor(buf195, (128, 1024), (1024, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg147_1, (1024, 1024), (1, 1024), 0), out=buf232)
        del arg147_1
        buf233 = reinterpret_tensor(buf211, (128, 1024), (1024, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 1024), (1, 1024), 0), out=buf233)
        del arg149_1
        buf234 = reinterpret_tensor(buf210, (128, 1024), (1024, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), out=buf234)
        del arg151_1
        buf235 = reinterpret_tensor(buf231, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf232, arg148_1, buf235, 131072, grid=grid(131072), stream=stream0)
        del arg148_1
        buf236 = reinterpret_tensor(buf232, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf233, arg150_1, buf236, 131072, grid=grid(131072), stream=stream0)
        del arg150_1
        buf237 = reinterpret_tensor(buf233, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf234, arg152_1, buf237, 131072, grid=grid(131072), stream=stream0)
        del arg152_1
        # Source Nodes: [], Original ATen: []
        buf238 = aten._scaled_dot_product_efficient_attention(buf235, buf236, buf237, None, True, scale=1.0)
        buf239 = buf238[0]
        del buf238
        buf243 = reinterpret_tensor(buf239, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf239  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf243, 131072, grid=grid(131072), stream=stream0)
        buf244 = reinterpret_tensor(buf237, (128, 1024), (1024, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), out=buf244)
        del arg153_1
        buf245 = reinterpret_tensor(buf244, (1, 128, 1024), (131072, 1024, 1), 0); del buf244  # reuse
        buf249 = reinterpret_tensor(buf243, (1, 128, 1024), (131072, 1024, 1), 0); del buf243  # reuse
        # Source Nodes: [hidden_states_105, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf245, buf220, buf227, arg144_1, arg154_1, arg155_1, arg156_1, buf249, 128, 1024, grid=grid(128), stream=stream0)
        del arg144_1
        del arg154_1
        del arg155_1
        del arg156_1
        buf250 = reinterpret_tensor(buf226, (128, 4096), (4096, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg157_1, (1024, 4096), (1, 1024), 0), out=buf250)
        del arg157_1
        buf251 = reinterpret_tensor(buf250, (1, 128, 4096), (524288, 4096, 1), 0); del buf250  # reuse
        # Source Nodes: [hidden_states_106], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf251, arg158_1, 524288, grid=grid(524288), stream=stream0)
        del arg158_1
        buf252 = reinterpret_tensor(buf249, (128, 1024), (1024, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg159_1, (4096, 1024), (1, 4096), 0), out=buf252)
        del arg159_1
        buf256 = reinterpret_tensor(buf227, (1, 128, 1024), (131072, 1024, 1), 0); del buf227  # reuse
        # Source Nodes: [hidden_states_112, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf245, buf252, arg160_1, arg161_1, arg162_1, buf256, 128, 1024, grid=grid(128), stream=stream0)
        del arg161_1
        del arg162_1
        buf257 = reinterpret_tensor(buf220, (128, 1024), (1024, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg163_1, (1024, 1024), (1, 1024), 0), out=buf257)
        del arg163_1
        buf258 = reinterpret_tensor(buf236, (128, 1024), (1024, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 1024), (1, 1024), 0), out=buf258)
        del arg165_1
        buf259 = reinterpret_tensor(buf235, (128, 1024), (1024, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), out=buf259)
        del arg167_1
        buf260 = reinterpret_tensor(buf256, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf257, arg164_1, buf260, 131072, grid=grid(131072), stream=stream0)
        del arg164_1
        buf261 = reinterpret_tensor(buf257, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf258, arg166_1, buf261, 131072, grid=grid(131072), stream=stream0)
        del arg166_1
        buf262 = reinterpret_tensor(buf258, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf259, arg168_1, buf262, 131072, grid=grid(131072), stream=stream0)
        del arg168_1
        # Source Nodes: [], Original ATen: []
        buf263 = aten._scaled_dot_product_efficient_attention(buf260, buf261, buf262, None, True, scale=1.0)
        buf264 = buf263[0]
        del buf263
        buf268 = reinterpret_tensor(buf264, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf264  # reuse
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf268, 131072, grid=grid(131072), stream=stream0)
        buf269 = reinterpret_tensor(buf262, (128, 1024), (1024, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), out=buf269)
        del arg169_1
        buf270 = reinterpret_tensor(buf269, (1, 128, 1024), (131072, 1024, 1), 0); del buf269  # reuse
        buf274 = reinterpret_tensor(buf268, (1, 128, 1024), (131072, 1024, 1), 0); del buf268  # reuse
        # Source Nodes: [hidden_states_116, residual_20, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf270, buf245, buf252, arg160_1, arg170_1, arg171_1, arg172_1, buf274, 128, 1024, grid=grid(128), stream=stream0)
        del arg160_1
        del arg170_1
        del arg171_1
        del arg172_1
        buf275 = reinterpret_tensor(buf251, (128, 4096), (4096, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg173_1, (1024, 4096), (1, 1024), 0), out=buf275)
        del arg173_1
        buf276 = reinterpret_tensor(buf275, (1, 128, 4096), (524288, 4096, 1), 0); del buf275  # reuse
        # Source Nodes: [hidden_states_117], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf276, arg174_1, 524288, grid=grid(524288), stream=stream0)
        del arg174_1
        buf277 = reinterpret_tensor(buf274, (128, 1024), (1024, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg175_1, (4096, 1024), (1, 4096), 0), out=buf277)
        del arg175_1
        buf281 = reinterpret_tensor(buf252, (1, 128, 1024), (131072, 1024, 1), 0); del buf252  # reuse
        # Source Nodes: [hidden_states_123, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf270, buf277, arg176_1, arg177_1, arg178_1, buf281, 128, 1024, grid=grid(128), stream=stream0)
        del arg177_1
        del arg178_1
        buf282 = reinterpret_tensor(buf245, (128, 1024), (1024, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg179_1, (1024, 1024), (1, 1024), 0), out=buf282)
        del arg179_1
        buf283 = reinterpret_tensor(buf261, (128, 1024), (1024, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 1024), (1, 1024), 0), out=buf283)
        del arg181_1
        buf284 = reinterpret_tensor(buf260, (128, 1024), (1024, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), out=buf284)
        del arg183_1
        buf285 = reinterpret_tensor(buf281, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf282, arg180_1, buf285, 131072, grid=grid(131072), stream=stream0)
        del arg180_1
        buf286 = reinterpret_tensor(buf282, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf283, arg182_1, buf286, 131072, grid=grid(131072), stream=stream0)
        del arg182_1
        buf287 = reinterpret_tensor(buf283, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf284, arg184_1, buf287, 131072, grid=grid(131072), stream=stream0)
        del arg184_1
        # Source Nodes: [], Original ATen: []
        buf288 = aten._scaled_dot_product_efficient_attention(buf285, buf286, buf287, None, True, scale=1.0)
        buf289 = buf288[0]
        del buf288
        buf293 = reinterpret_tensor(buf289, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf289  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf293, 131072, grid=grid(131072), stream=stream0)
        buf294 = reinterpret_tensor(buf287, (128, 1024), (1024, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), out=buf294)
        del arg185_1
        buf295 = reinterpret_tensor(buf294, (1, 128, 1024), (131072, 1024, 1), 0); del buf294  # reuse
        buf299 = reinterpret_tensor(buf293, (1, 128, 1024), (131072, 1024, 1), 0); del buf293  # reuse
        # Source Nodes: [hidden_states_127, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf295, buf270, buf277, arg176_1, arg186_1, arg187_1, arg188_1, buf299, 128, 1024, grid=grid(128), stream=stream0)
        del arg176_1
        del arg186_1
        del arg187_1
        del arg188_1
        buf300 = reinterpret_tensor(buf276, (128, 4096), (4096, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg189_1, (1024, 4096), (1, 1024), 0), out=buf300)
        del arg189_1
        buf301 = reinterpret_tensor(buf300, (1, 128, 4096), (524288, 4096, 1), 0); del buf300  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf301, arg190_1, 524288, grid=grid(524288), stream=stream0)
        del arg190_1
        buf302 = reinterpret_tensor(buf299, (128, 1024), (1024, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf301, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg191_1, (4096, 1024), (1, 4096), 0), out=buf302)
        del arg191_1
        buf332 = reinterpret_tensor(buf277, (1, 128, 1024), (131072, 1024, 1), 0); del buf277  # reuse
        # Source Nodes: [hidden_states_133, hidden_states_134], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf295, buf302, arg192_1, arg193_1, arg194_1, buf332, 128, 1024, grid=grid(128), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        buf306 = buf0; del buf0  # reuse
        # Source Nodes: [cumsum_1, mask_3, ne_1], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        triton_poi_fused__to_copy_cumsum_ne_0.run(arg514_1, buf306, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [cumsum_1, mask_3, ne_1], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf307 = aten.cumsum(buf306, 1)
        del buf306
        buf308 = buf307
        del buf307
        buf312 = reinterpret_tensor(buf302, (1, 128, 1024), (131072, 1024, 1), 0); del buf302  # reuse
        # Source Nodes: [hidden_states_135, hidden_states_137, inputs_embeds_1, l__mod___model_decoder_embed_tokens], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_embedding_mul_native_layer_norm_1.run(arg514_1, arg195_1, buf308, arg512_1, arg196_1, arg197_1, buf312, 128, 1024, grid=grid(128), stream=stream0)
        del arg196_1
        del arg197_1
        buf313 = reinterpret_tensor(buf295, (128, 1024), (1024, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg198_1, (1024, 1024), (1, 1024), 0), out=buf313)
        del arg198_1
        buf314 = reinterpret_tensor(buf270, (128, 1024), (1024, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg200_1, (1024, 1024), (1, 1024), 0), out=buf314)
        del arg200_1
        buf315 = buf286; del buf286  # reuse
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf314, arg201_1, buf315, 131072, grid=grid(131072), stream=stream0)
        del arg201_1
        buf316 = reinterpret_tensor(buf314, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf314  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf313, arg199_1, buf316, 131072, grid=grid(131072), stream=stream0)
        del arg199_1
        buf317 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf315, (16, 64, 128), (8192, 1, 64), 0), out=buf317)
        buf322 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf317, buf322, 2048, 128, grid=grid(2048), stream=stream0)
        buf320 = reinterpret_tensor(buf316, (128, 1024), (1024, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg202_1, (1024, 1024), (1, 1024), 0), out=buf320)
        del arg202_1
        buf321 = reinterpret_tensor(buf312, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf312  # reuse
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf320, arg203_1, buf321, 131072, grid=grid(131072), stream=stream0)
        del arg203_1
        buf323 = reinterpret_tensor(buf320, (16, 128, 64), (8192, 64, 1), 0); del buf320  # reuse
        # Source Nodes: [attn_output_60, attn_weights_27], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf322, reinterpret_tensor(buf321, (16, 128, 64), (8192, 64, 1), 0), out=buf323)
        buf324 = reinterpret_tensor(buf313, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf313  # reuse
        # Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf323, buf324, 131072, grid=grid(131072), stream=stream0)
        buf325 = reinterpret_tensor(buf323, (128, 1024), (1024, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg204_1, (1024, 1024), (1, 1024), 0), out=buf325)
        del arg204_1
        buf326 = reinterpret_tensor(buf325, (1, 128, 1024), (131072, 1024, 1), 0); del buf325  # reuse
        buf330 = reinterpret_tensor(buf324, (1, 128, 1024), (131072, 1024, 1), 0); del buf324  # reuse
        # Source Nodes: [hidden_states_135, hidden_states_141, inputs_embeds_1, l__mod___model_decoder_embed_tokens, residual_25], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_embedding_mul_native_layer_norm_5.run(buf326, arg514_1, arg195_1, buf308, arg512_1, arg205_1, arg206_1, arg207_1, buf330, 128, 1024, grid=grid(128), stream=stream0)
        del arg195_1
        del arg205_1
        del arg206_1
        del arg207_1
        del arg512_1
        del arg514_1
        del buf308
        buf331 = reinterpret_tensor(buf285, (128, 1024), (1024, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf330, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg208_1, (1024, 1024), (1, 1024), 0), out=buf331)
        del arg208_1
        buf333 = reinterpret_tensor(buf330, (128, 1024), (1024, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg210_1, (1024, 1024), (1, 1024), 0), out=buf333)
        del arg210_1
        buf334 = reinterpret_tensor(buf284, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf284  # reuse
        # Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf333, arg211_1, buf334, 131072, grid=grid(131072), stream=stream0)
        del arg211_1
        buf335 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg212_1, (1024, 1024), (1, 1024), 0), out=buf335)
        del arg212_1
        buf336 = reinterpret_tensor(buf259, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf335, arg213_1, buf336, 131072, grid=grid(131072), stream=stream0)
        del arg213_1
        buf337 = reinterpret_tensor(buf335, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf331, arg209_1, buf337, 131072, grid=grid(131072), stream=stream0)
        del arg209_1
        # Source Nodes: [], Original ATen: []
        buf338 = aten._scaled_dot_product_efficient_attention(buf337, reinterpret_tensor(buf334, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf336, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf339 = buf338[0]
        del buf338
        buf343 = reinterpret_tensor(buf339, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf339  # reuse
        # Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf343, 131072, grid=grid(131072), stream=stream0)
        buf344 = reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg214_1, (1024, 1024), (1, 1024), 0), out=buf344)
        del arg214_1
        buf348 = reinterpret_tensor(buf343, (1, 128, 1024), (131072, 1024, 1), 0); del buf343  # reuse
        # Source Nodes: [hidden_states_145, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf326, buf344, arg215_1, arg216_1, arg217_1, buf348, 128, 1024, grid=grid(128), stream=stream0)
        del arg216_1
        del arg217_1
        buf349 = reinterpret_tensor(buf301, (128, 4096), (4096, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg218_1, (1024, 4096), (1, 1024), 0), out=buf349)
        del arg218_1
        buf350 = reinterpret_tensor(buf349, (1, 128, 4096), (524288, 4096, 1), 0); del buf349  # reuse
        # Source Nodes: [hidden_states_146], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf350, arg219_1, 524288, grid=grid(524288), stream=stream0)
        del arg219_1
        buf351 = reinterpret_tensor(buf348, (128, 1024), (1024, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg220_1, (4096, 1024), (1, 4096), 0), out=buf351)
        del arg220_1
        buf352 = reinterpret_tensor(buf351, (1, 128, 1024), (131072, 1024, 1), 0); del buf351  # reuse
        buf356 = reinterpret_tensor(buf331, (1, 128, 1024), (131072, 1024, 1), 0); del buf331  # reuse
        # Source Nodes: [hidden_states_152, residual_26, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf352, buf326, buf344, arg215_1, arg221_1, arg222_1, arg223_1, buf356, 128, 1024, grid=grid(128), stream=stream0)
        del arg215_1
        del arg221_1
        del arg222_1
        del arg223_1
        buf357 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg224_1, (1024, 1024), (1, 1024), 0), out=buf357)
        del arg224_1
        buf358 = reinterpret_tensor(buf326, (128, 1024), (1024, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg226_1, (1024, 1024), (1, 1024), 0), out=buf358)
        del arg226_1
        buf359 = reinterpret_tensor(buf234, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf234  # reuse
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf358, arg227_1, buf359, 131072, grid=grid(131072), stream=stream0)
        del arg227_1
        buf360 = reinterpret_tensor(buf358, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf358  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf357, arg225_1, buf360, 131072, grid=grid(131072), stream=stream0)
        del arg225_1
        buf361 = buf322; del buf322  # reuse
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf359, (16, 64, 128), (8192, 1, 64), 0), out=buf361)
        buf366 = buf317; del buf317  # reuse
        # Source Nodes: [attn_weights_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf361, buf366, 2048, 128, grid=grid(2048), stream=stream0)
        buf364 = reinterpret_tensor(buf360, (128, 1024), (1024, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg228_1, (1024, 1024), (1, 1024), 0), out=buf364)
        del arg228_1
        buf365 = reinterpret_tensor(buf356, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf356  # reuse
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf364, arg229_1, buf365, 131072, grid=grid(131072), stream=stream0)
        del arg229_1
        buf367 = reinterpret_tensor(buf364, (16, 128, 64), (8192, 64, 1), 0); del buf364  # reuse
        # Source Nodes: [attn_output_70, attn_weights_33], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf366, reinterpret_tensor(buf365, (16, 128, 64), (8192, 64, 1), 0), out=buf367)
        buf368 = reinterpret_tensor(buf357, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf357  # reuse
        # Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf367, buf368, 131072, grid=grid(131072), stream=stream0)
        buf369 = reinterpret_tensor(buf367, (128, 1024), (1024, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf368, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg230_1, (1024, 1024), (1, 1024), 0), out=buf369)
        del arg230_1
        buf373 = reinterpret_tensor(buf368, (1, 128, 1024), (131072, 1024, 1), 0); del buf368  # reuse
        # Source Nodes: [hidden_states_156, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf352, buf369, arg231_1, arg232_1, arg233_1, buf373, 128, 1024, grid=grid(128), stream=stream0)
        del arg232_1
        del arg233_1
        buf374 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg234_1, (1024, 1024), (1, 1024), 0), out=buf374)
        del arg234_1
        buf375 = reinterpret_tensor(buf373, (128, 1024), (1024, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg236_1, (1024, 1024), (1, 1024), 0), out=buf375)
        del arg236_1
        buf376 = reinterpret_tensor(buf184, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf184  # reuse
        # Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf375, arg237_1, buf376, 131072, grid=grid(131072), stream=stream0)
        del arg237_1
        buf377 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg238_1, (1024, 1024), (1, 1024), 0), out=buf377)
        del arg238_1
        buf378 = reinterpret_tensor(buf159, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf159  # reuse
        # Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf377, arg239_1, buf378, 131072, grid=grid(131072), stream=stream0)
        del arg239_1
        buf379 = reinterpret_tensor(buf377, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf374, arg235_1, buf379, 131072, grid=grid(131072), stream=stream0)
        del arg235_1
        # Source Nodes: [], Original ATen: []
        buf380 = aten._scaled_dot_product_efficient_attention(buf379, reinterpret_tensor(buf376, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf378, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf381 = buf380[0]
        del buf380
        buf385 = reinterpret_tensor(buf381, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf381  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf385, 131072, grid=grid(131072), stream=stream0)
        buf386 = reinterpret_tensor(buf379, (128, 1024), (1024, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf385, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg240_1, (1024, 1024), (1, 1024), 0), out=buf386)
        del arg240_1
        buf387 = reinterpret_tensor(buf386, (1, 128, 1024), (131072, 1024, 1), 0); del buf386  # reuse
        buf391 = reinterpret_tensor(buf385, (1, 128, 1024), (131072, 1024, 1), 0); del buf385  # reuse
        # Source Nodes: [hidden_states_160, residual_28, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf387, buf352, buf369, arg231_1, arg241_1, arg242_1, arg243_1, buf391, 128, 1024, grid=grid(128), stream=stream0)
        del arg231_1
        del arg241_1
        del arg242_1
        del arg243_1
        buf392 = reinterpret_tensor(buf350, (128, 4096), (4096, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf391, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg244_1, (1024, 4096), (1, 1024), 0), out=buf392)
        del arg244_1
        buf393 = reinterpret_tensor(buf392, (1, 128, 4096), (524288, 4096, 1), 0); del buf392  # reuse
        # Source Nodes: [hidden_states_161], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf393, arg245_1, 524288, grid=grid(524288), stream=stream0)
        del arg245_1
        buf394 = reinterpret_tensor(buf391, (128, 1024), (1024, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf393, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg246_1, (4096, 1024), (1, 4096), 0), out=buf394)
        del arg246_1
        buf398 = reinterpret_tensor(buf369, (1, 128, 1024), (131072, 1024, 1), 0); del buf369  # reuse
        # Source Nodes: [hidden_states_167, residual_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf387, buf394, arg247_1, arg248_1, arg249_1, buf398, 128, 1024, grid=grid(128), stream=stream0)
        del arg248_1
        del arg249_1
        buf399 = reinterpret_tensor(buf352, (128, 1024), (1024, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg250_1, (1024, 1024), (1, 1024), 0), out=buf399)
        del arg250_1
        buf400 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg252_1, (1024, 1024), (1, 1024), 0), out=buf400)
        del arg252_1
        buf401 = reinterpret_tensor(buf134, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf134  # reuse
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf400, arg253_1, buf401, 131072, grid=grid(131072), stream=stream0)
        del arg253_1
        buf402 = reinterpret_tensor(buf400, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf400  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf399, arg251_1, buf402, 131072, grid=grid(131072), stream=stream0)
        del arg251_1
        buf403 = buf366; del buf366  # reuse
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf402, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf401, (16, 64, 128), (8192, 1, 64), 0), out=buf403)
        buf408 = buf361; del buf361  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf403, buf408, 2048, 128, grid=grid(2048), stream=stream0)
        buf406 = reinterpret_tensor(buf402, (128, 1024), (1024, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg254_1, (1024, 1024), (1, 1024), 0), out=buf406)
        del arg254_1
        buf407 = reinterpret_tensor(buf398, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf398  # reuse
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf406, arg255_1, buf407, 131072, grid=grid(131072), stream=stream0)
        del arg255_1
        buf409 = reinterpret_tensor(buf406, (16, 128, 64), (8192, 64, 1), 0); del buf406  # reuse
        # Source Nodes: [attn_output_80, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf408, reinterpret_tensor(buf407, (16, 128, 64), (8192, 64, 1), 0), out=buf409)
        buf410 = reinterpret_tensor(buf399, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf399  # reuse
        # Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf409, buf410, 131072, grid=grid(131072), stream=stream0)
        buf411 = reinterpret_tensor(buf409, (128, 1024), (1024, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg256_1, (1024, 1024), (1, 1024), 0), out=buf411)
        del arg256_1
        buf412 = reinterpret_tensor(buf411, (1, 128, 1024), (131072, 1024, 1), 0); del buf411  # reuse
        buf416 = reinterpret_tensor(buf410, (1, 128, 1024), (131072, 1024, 1), 0); del buf410  # reuse
        # Source Nodes: [hidden_states_171, residual_30, residual_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf412, buf387, buf394, arg247_1, arg257_1, arg258_1, arg259_1, buf416, 128, 1024, grid=grid(128), stream=stream0)
        del arg247_1
        del arg257_1
        del arg258_1
        del arg259_1
        buf417 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg260_1, (1024, 1024), (1, 1024), 0), out=buf417)
        del arg260_1
        buf418 = reinterpret_tensor(buf416, (128, 1024), (1024, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg262_1, (1024, 1024), (1, 1024), 0), out=buf418)
        del arg262_1
        buf419 = reinterpret_tensor(buf387, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf387  # reuse
        # Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf418, arg263_1, buf419, 131072, grid=grid(131072), stream=stream0)
        del arg263_1
        buf420 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg264_1, (1024, 1024), (1, 1024), 0), out=buf420)
        del arg264_1
        buf421 = reinterpret_tensor(buf109, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf109  # reuse
        # Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf420, arg265_1, buf421, 131072, grid=grid(131072), stream=stream0)
        del arg265_1
        buf422 = reinterpret_tensor(buf420, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf417, arg261_1, buf422, 131072, grid=grid(131072), stream=stream0)
        del arg261_1
        # Source Nodes: [], Original ATen: []
        buf423 = aten._scaled_dot_product_efficient_attention(buf422, reinterpret_tensor(buf419, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf421, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf424 = buf423[0]
        del buf423
        buf428 = reinterpret_tensor(buf424, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf424  # reuse
        # Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf428, 131072, grid=grid(131072), stream=stream0)
        buf429 = reinterpret_tensor(buf422, (128, 1024), (1024, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg266_1, (1024, 1024), (1, 1024), 0), out=buf429)
        del arg266_1
        buf433 = reinterpret_tensor(buf428, (1, 128, 1024), (131072, 1024, 1), 0); del buf428  # reuse
        # Source Nodes: [hidden_states_175, residual_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf412, buf429, arg267_1, arg268_1, arg269_1, buf433, 128, 1024, grid=grid(128), stream=stream0)
        del arg268_1
        del arg269_1
        buf434 = reinterpret_tensor(buf393, (128, 4096), (4096, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg270_1, (1024, 4096), (1, 1024), 0), out=buf434)
        del arg270_1
        buf435 = reinterpret_tensor(buf434, (1, 128, 4096), (524288, 4096, 1), 0); del buf434  # reuse
        # Source Nodes: [hidden_states_176], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf435, arg271_1, 524288, grid=grid(524288), stream=stream0)
        del arg271_1
        buf436 = reinterpret_tensor(buf433, (128, 1024), (1024, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg272_1, (4096, 1024), (1, 4096), 0), out=buf436)
        del arg272_1
        buf437 = reinterpret_tensor(buf436, (1, 128, 1024), (131072, 1024, 1), 0); del buf436  # reuse
        buf441 = reinterpret_tensor(buf417, (1, 128, 1024), (131072, 1024, 1), 0); del buf417  # reuse
        # Source Nodes: [hidden_states_182, residual_32, residual_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf437, buf412, buf429, arg267_1, arg273_1, arg274_1, arg275_1, buf441, 128, 1024, grid=grid(128), stream=stream0)
        del arg267_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf442 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg276_1, (1024, 1024), (1, 1024), 0), out=buf442)
        del arg276_1
        buf443 = reinterpret_tensor(buf412, (128, 1024), (1024, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg278_1, (1024, 1024), (1, 1024), 0), out=buf443)
        del arg278_1
        buf444 = reinterpret_tensor(buf84, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf84  # reuse
        # Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf443, arg279_1, buf444, 131072, grid=grid(131072), stream=stream0)
        del arg279_1
        buf445 = reinterpret_tensor(buf443, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf443  # reuse
        # Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf442, arg277_1, buf445, 131072, grid=grid(131072), stream=stream0)
        del arg277_1
        buf446 = buf408; del buf408  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf444, (16, 64, 128), (8192, 1, 64), 0), out=buf446)
        buf451 = buf403; del buf403  # reuse
        # Source Nodes: [attn_weights_45], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf446, buf451, 2048, 128, grid=grid(2048), stream=stream0)
        buf449 = reinterpret_tensor(buf445, (128, 1024), (1024, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg280_1, (1024, 1024), (1, 1024), 0), out=buf449)
        del arg280_1
        buf450 = reinterpret_tensor(buf441, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf441  # reuse
        # Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf449, arg281_1, buf450, 131072, grid=grid(131072), stream=stream0)
        del arg281_1
        buf452 = reinterpret_tensor(buf449, (16, 128, 64), (8192, 64, 1), 0); del buf449  # reuse
        # Source Nodes: [attn_output_90, attn_weights_45], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf451, reinterpret_tensor(buf450, (16, 128, 64), (8192, 64, 1), 0), out=buf452)
        buf453 = reinterpret_tensor(buf442, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf442  # reuse
        # Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf452, buf453, 131072, grid=grid(131072), stream=stream0)
        buf454 = reinterpret_tensor(buf452, (128, 1024), (1024, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg282_1, (1024, 1024), (1, 1024), 0), out=buf454)
        del arg282_1
        buf458 = reinterpret_tensor(buf453, (1, 128, 1024), (131072, 1024, 1), 0); del buf453  # reuse
        # Source Nodes: [hidden_states_186, residual_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf437, buf454, arg283_1, arg284_1, arg285_1, buf458, 128, 1024, grid=grid(128), stream=stream0)
        del arg284_1
        del arg285_1
        buf459 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg286_1, (1024, 1024), (1, 1024), 0), out=buf459)
        del arg286_1
        buf460 = reinterpret_tensor(buf458, (128, 1024), (1024, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg288_1, (1024, 1024), (1, 1024), 0), out=buf460)
        del arg288_1
        buf461 = reinterpret_tensor(buf34, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [key_states_38], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf460, arg289_1, buf461, 131072, grid=grid(131072), stream=stream0)
        del arg289_1
        buf462 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg290_1, (1024, 1024), (1, 1024), 0), out=buf462)
        del arg290_1
        buf463 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_38], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf462, arg291_1, buf463, 131072, grid=grid(131072), stream=stream0)
        del arg291_1
        buf464 = reinterpret_tensor(buf462, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf459, arg287_1, buf464, 131072, grid=grid(131072), stream=stream0)
        del arg287_1
        # Source Nodes: [], Original ATen: []
        buf465 = aten._scaled_dot_product_efficient_attention(buf464, reinterpret_tensor(buf461, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf463, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf466 = buf465[0]
        del buf465
        buf470 = reinterpret_tensor(buf466, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf466  # reuse
        # Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf470, 131072, grid=grid(131072), stream=stream0)
        buf471 = reinterpret_tensor(buf464, (128, 1024), (1024, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf470, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg292_1, (1024, 1024), (1, 1024), 0), out=buf471)
        del arg292_1
        buf472 = reinterpret_tensor(buf471, (1, 128, 1024), (131072, 1024, 1), 0); del buf471  # reuse
        buf476 = reinterpret_tensor(buf470, (1, 128, 1024), (131072, 1024, 1), 0); del buf470  # reuse
        # Source Nodes: [hidden_states_190, residual_34, residual_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf472, buf437, buf454, arg283_1, arg293_1, arg294_1, arg295_1, buf476, 128, 1024, grid=grid(128), stream=stream0)
        del arg283_1
        del arg293_1
        del arg294_1
        del arg295_1
        buf477 = reinterpret_tensor(buf435, (128, 4096), (4096, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf476, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg296_1, (1024, 4096), (1, 1024), 0), out=buf477)
        del arg296_1
        buf478 = reinterpret_tensor(buf477, (1, 128, 4096), (524288, 4096, 1), 0); del buf477  # reuse
        # Source Nodes: [hidden_states_191], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf478, arg297_1, 524288, grid=grid(524288), stream=stream0)
        del arg297_1
        buf479 = reinterpret_tensor(buf476, (128, 1024), (1024, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg298_1, (4096, 1024), (1, 4096), 0), out=buf479)
        del arg298_1
        buf483 = reinterpret_tensor(buf454, (1, 128, 1024), (131072, 1024, 1), 0); del buf454  # reuse
        # Source Nodes: [hidden_states_197, residual_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf472, buf479, arg299_1, arg300_1, arg301_1, buf483, 128, 1024, grid=grid(128), stream=stream0)
        del arg300_1
        del arg301_1
        buf484 = reinterpret_tensor(buf437, (128, 1024), (1024, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf483, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg302_1, (1024, 1024), (1, 1024), 0), out=buf484)
        del arg302_1
        buf485 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf483, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg304_1, (1024, 1024), (1, 1024), 0), out=buf485)
        del arg304_1
        buf486 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf485, arg305_1, buf486, 131072, grid=grid(131072), stream=stream0)
        del arg305_1
        buf487 = reinterpret_tensor(buf485, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf485  # reuse
        # Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf484, arg303_1, buf487, 131072, grid=grid(131072), stream=stream0)
        del arg303_1
        buf488 = buf451; del buf451  # reuse
        # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf487, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf486, (16, 64, 128), (8192, 1, 64), 0), out=buf488)
        buf493 = buf446; del buf446  # reuse
        # Source Nodes: [attn_weights_51], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf488, buf493, 2048, 128, grid=grid(2048), stream=stream0)
        buf491 = reinterpret_tensor(buf487, (128, 1024), (1024, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf483, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg306_1, (1024, 1024), (1, 1024), 0), out=buf491)
        del arg306_1
        buf492 = reinterpret_tensor(buf483, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf483  # reuse
        # Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf491, arg307_1, buf492, 131072, grid=grid(131072), stream=stream0)
        del arg307_1
        buf494 = reinterpret_tensor(buf491, (16, 128, 64), (8192, 64, 1), 0); del buf491  # reuse
        # Source Nodes: [attn_output_100, attn_weights_51], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf493, reinterpret_tensor(buf492, (16, 128, 64), (8192, 64, 1), 0), out=buf494)
        buf495 = reinterpret_tensor(buf484, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf484  # reuse
        # Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf494, buf495, 131072, grid=grid(131072), stream=stream0)
        buf496 = reinterpret_tensor(buf494, (128, 1024), (1024, 1), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf495, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg308_1, (1024, 1024), (1, 1024), 0), out=buf496)
        del arg308_1
        buf497 = reinterpret_tensor(buf496, (1, 128, 1024), (131072, 1024, 1), 0); del buf496  # reuse
        buf501 = reinterpret_tensor(buf495, (1, 128, 1024), (131072, 1024, 1), 0); del buf495  # reuse
        # Source Nodes: [hidden_states_201, residual_36, residual_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf497, buf472, buf479, arg299_1, arg309_1, arg310_1, arg311_1, buf501, 128, 1024, grid=grid(128), stream=stream0)
        del arg299_1
        del arg309_1
        del arg310_1
        del arg311_1
        buf502 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf501, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg312_1, (1024, 1024), (1, 1024), 0), out=buf502)
        del arg312_1
        buf503 = reinterpret_tensor(buf501, (128, 1024), (1024, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg314_1, (1024, 1024), (1, 1024), 0), out=buf503)
        del arg314_1
        buf504 = reinterpret_tensor(buf472, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf472  # reuse
        # Source Nodes: [key_states_42], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf503, arg315_1, buf504, 131072, grid=grid(131072), stream=stream0)
        del arg315_1
        buf505 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg316_1, (1024, 1024), (1, 1024), 0), out=buf505)
        del arg316_1
        buf506 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_42], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf505, arg317_1, buf506, 131072, grid=grid(131072), stream=stream0)
        del arg317_1
        buf507 = reinterpret_tensor(buf505, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf502, arg313_1, buf507, 131072, grid=grid(131072), stream=stream0)
        del arg313_1
        # Source Nodes: [], Original ATen: []
        buf508 = aten._scaled_dot_product_efficient_attention(buf507, reinterpret_tensor(buf504, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf506, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf509 = buf508[0]
        del buf508
        buf513 = reinterpret_tensor(buf509, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf509  # reuse
        # Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf513, 131072, grid=grid(131072), stream=stream0)
        buf514 = reinterpret_tensor(buf507, (128, 1024), (1024, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf513, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg318_1, (1024, 1024), (1, 1024), 0), out=buf514)
        del arg318_1
        buf518 = reinterpret_tensor(buf513, (1, 128, 1024), (131072, 1024, 1), 0); del buf513  # reuse
        # Source Nodes: [hidden_states_205, residual_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf497, buf514, arg319_1, arg320_1, arg321_1, buf518, 128, 1024, grid=grid(128), stream=stream0)
        del arg320_1
        del arg321_1
        buf519 = reinterpret_tensor(buf478, (128, 4096), (4096, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf518, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg322_1, (1024, 4096), (1, 1024), 0), out=buf519)
        del arg322_1
        buf520 = reinterpret_tensor(buf519, (1, 128, 4096), (524288, 4096, 1), 0); del buf519  # reuse
        # Source Nodes: [hidden_states_206], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf520, arg323_1, 524288, grid=grid(524288), stream=stream0)
        del arg323_1
        buf521 = reinterpret_tensor(buf518, (128, 1024), (1024, 1), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg324_1, (4096, 1024), (1, 4096), 0), out=buf521)
        del arg324_1
        buf522 = reinterpret_tensor(buf521, (1, 128, 1024), (131072, 1024, 1), 0); del buf521  # reuse
        buf526 = reinterpret_tensor(buf502, (1, 128, 1024), (131072, 1024, 1), 0); del buf502  # reuse
        # Source Nodes: [hidden_states_212, residual_38, residual_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf522, buf497, buf514, arg319_1, arg325_1, arg326_1, arg327_1, buf526, 128, 1024, grid=grid(128), stream=stream0)
        del arg319_1
        del arg325_1
        del arg326_1
        del arg327_1
        buf527 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf526, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg328_1, (1024, 1024), (1, 1024), 0), out=buf527)
        del arg328_1
        buf528 = reinterpret_tensor(buf497, (128, 1024), (1024, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf526, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg330_1, (1024, 1024), (1, 1024), 0), out=buf528)
        del arg330_1
        buf529 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf528, arg331_1, buf529, 131072, grid=grid(131072), stream=stream0)
        del arg331_1
        buf530 = reinterpret_tensor(buf528, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf528  # reuse
        # Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf527, arg329_1, buf530, 131072, grid=grid(131072), stream=stream0)
        del arg329_1
        buf531 = buf493; del buf493  # reuse
        # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf530, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf529, (16, 64, 128), (8192, 1, 64), 0), out=buf531)
        buf536 = buf488; del buf488  # reuse
        # Source Nodes: [attn_weights_57], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf531, buf536, 2048, 128, grid=grid(2048), stream=stream0)
        buf534 = reinterpret_tensor(buf530, (128, 1024), (1024, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf526, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg332_1, (1024, 1024), (1, 1024), 0), out=buf534)
        del arg332_1
        buf535 = reinterpret_tensor(buf526, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf526  # reuse
        # Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf534, arg333_1, buf535, 131072, grid=grid(131072), stream=stream0)
        del arg333_1
        buf537 = reinterpret_tensor(buf534, (16, 128, 64), (8192, 64, 1), 0); del buf534  # reuse
        # Source Nodes: [attn_output_110, attn_weights_57], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf536, reinterpret_tensor(buf535, (16, 128, 64), (8192, 64, 1), 0), out=buf537)
        buf538 = reinterpret_tensor(buf527, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf527  # reuse
        # Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf537, buf538, 131072, grid=grid(131072), stream=stream0)
        buf539 = reinterpret_tensor(buf537, (128, 1024), (1024, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf538, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg334_1, (1024, 1024), (1, 1024), 0), out=buf539)
        del arg334_1
        buf543 = reinterpret_tensor(buf538, (1, 128, 1024), (131072, 1024, 1), 0); del buf538  # reuse
        # Source Nodes: [hidden_states_216, residual_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf522, buf539, arg335_1, arg336_1, arg337_1, buf543, 128, 1024, grid=grid(128), stream=stream0)
        del arg336_1
        del arg337_1
        buf544 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf543, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg338_1, (1024, 1024), (1, 1024), 0), out=buf544)
        del arg338_1
        buf545 = reinterpret_tensor(buf543, (128, 1024), (1024, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg340_1, (1024, 1024), (1, 1024), 0), out=buf545)
        del arg340_1
        buf546 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_46], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf545, arg341_1, buf546, 131072, grid=grid(131072), stream=stream0)
        del arg341_1
        buf547 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg342_1, (1024, 1024), (1, 1024), 0), out=buf547)
        del arg342_1
        buf548 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_46], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf547, arg343_1, buf548, 131072, grid=grid(131072), stream=stream0)
        del arg343_1
        buf549 = reinterpret_tensor(buf547, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf544, arg339_1, buf549, 131072, grid=grid(131072), stream=stream0)
        del arg339_1
        # Source Nodes: [], Original ATen: []
        buf550 = aten._scaled_dot_product_efficient_attention(buf549, reinterpret_tensor(buf546, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf548, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf551 = buf550[0]
        del buf550
        buf555 = reinterpret_tensor(buf551, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf551  # reuse
        # Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf555, 131072, grid=grid(131072), stream=stream0)
        buf556 = reinterpret_tensor(buf549, (128, 1024), (1024, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg344_1, (1024, 1024), (1, 1024), 0), out=buf556)
        del arg344_1
        buf557 = reinterpret_tensor(buf556, (1, 128, 1024), (131072, 1024, 1), 0); del buf556  # reuse
        buf561 = reinterpret_tensor(buf555, (1, 128, 1024), (131072, 1024, 1), 0); del buf555  # reuse
        # Source Nodes: [hidden_states_220, residual_40, residual_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf557, buf522, buf539, arg335_1, arg345_1, arg346_1, arg347_1, buf561, 128, 1024, grid=grid(128), stream=stream0)
        del arg335_1
        del arg345_1
        del arg346_1
        del arg347_1
        buf562 = reinterpret_tensor(buf520, (128, 4096), (4096, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf561, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg348_1, (1024, 4096), (1, 1024), 0), out=buf562)
        del arg348_1
        buf563 = reinterpret_tensor(buf562, (1, 128, 4096), (524288, 4096, 1), 0); del buf562  # reuse
        # Source Nodes: [hidden_states_221], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf563, arg349_1, 524288, grid=grid(524288), stream=stream0)
        del arg349_1
        buf564 = reinterpret_tensor(buf561, (128, 1024), (1024, 1), 0); del buf561  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf563, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg350_1, (4096, 1024), (1, 4096), 0), out=buf564)
        del arg350_1
        buf568 = reinterpret_tensor(buf539, (1, 128, 1024), (131072, 1024, 1), 0); del buf539  # reuse
        # Source Nodes: [hidden_states_227, residual_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf557, buf564, arg351_1, arg352_1, arg353_1, buf568, 128, 1024, grid=grid(128), stream=stream0)
        del arg352_1
        del arg353_1
        buf569 = reinterpret_tensor(buf522, (128, 1024), (1024, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf568, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg354_1, (1024, 1024), (1, 1024), 0), out=buf569)
        del arg354_1
        buf570 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf568, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg356_1, (1024, 1024), (1, 1024), 0), out=buf570)
        del arg356_1
        buf571 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_48], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf570, arg357_1, buf571, 131072, grid=grid(131072), stream=stream0)
        del arg357_1
        buf572 = reinterpret_tensor(buf570, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf570  # reuse
        # Source Nodes: [contiguous_74], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf569, arg355_1, buf572, 131072, grid=grid(131072), stream=stream0)
        del arg355_1
        buf573 = buf536; del buf536  # reuse
        # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf572, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf571, (16, 64, 128), (8192, 1, 64), 0), out=buf573)
        buf578 = buf531; del buf531  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf573, buf578, 2048, 128, grid=grid(2048), stream=stream0)
        buf576 = reinterpret_tensor(buf572, (128, 1024), (1024, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf568, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg358_1, (1024, 1024), (1, 1024), 0), out=buf576)
        del arg358_1
        buf577 = reinterpret_tensor(buf568, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf568  # reuse
        # Source Nodes: [value_states_48], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf576, arg359_1, buf577, 131072, grid=grid(131072), stream=stream0)
        del arg359_1
        buf579 = reinterpret_tensor(buf576, (16, 128, 64), (8192, 64, 1), 0); del buf576  # reuse
        # Source Nodes: [attn_output_120, attn_weights_63], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf578, reinterpret_tensor(buf577, (16, 128, 64), (8192, 64, 1), 0), out=buf579)
        buf580 = reinterpret_tensor(buf569, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf569  # reuse
        # Source Nodes: [attn_output_123], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf579, buf580, 131072, grid=grid(131072), stream=stream0)
        buf581 = reinterpret_tensor(buf579, (128, 1024), (1024, 1), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg360_1, (1024, 1024), (1, 1024), 0), out=buf581)
        del arg360_1
        buf582 = reinterpret_tensor(buf581, (1, 128, 1024), (131072, 1024, 1), 0); del buf581  # reuse
        buf586 = reinterpret_tensor(buf580, (1, 128, 1024), (131072, 1024, 1), 0); del buf580  # reuse
        # Source Nodes: [hidden_states_231, residual_42, residual_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf582, buf557, buf564, arg351_1, arg361_1, arg362_1, arg363_1, buf586, 128, 1024, grid=grid(128), stream=stream0)
        del arg351_1
        del arg361_1
        del arg362_1
        del arg363_1
        buf587 = buf564; del buf564  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf586, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg364_1, (1024, 1024), (1, 1024), 0), out=buf587)
        del arg364_1
        buf588 = reinterpret_tensor(buf586, (128, 1024), (1024, 1), 0); del buf586  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg366_1, (1024, 1024), (1, 1024), 0), out=buf588)
        del arg366_1
        buf589 = reinterpret_tensor(buf557, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf557  # reuse
        # Source Nodes: [key_states_50], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf588, arg367_1, buf589, 131072, grid=grid(131072), stream=stream0)
        del arg367_1
        buf590 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg368_1, (1024, 1024), (1, 1024), 0), out=buf590)
        del arg368_1
        buf591 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_50], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf590, arg369_1, buf591, 131072, grid=grid(131072), stream=stream0)
        del arg369_1
        buf592 = reinterpret_tensor(buf590, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf587, arg365_1, buf592, 131072, grid=grid(131072), stream=stream0)
        del arg365_1
        # Source Nodes: [], Original ATen: []
        buf593 = aten._scaled_dot_product_efficient_attention(buf592, reinterpret_tensor(buf589, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf591, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf594 = buf593[0]
        del buf593
        buf598 = reinterpret_tensor(buf594, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf594  # reuse
        # Source Nodes: [attn_output_128], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf598, 131072, grid=grid(131072), stream=stream0)
        buf599 = reinterpret_tensor(buf592, (128, 1024), (1024, 1), 0); del buf592  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg370_1, (1024, 1024), (1, 1024), 0), out=buf599)
        del arg370_1
        buf603 = reinterpret_tensor(buf598, (1, 128, 1024), (131072, 1024, 1), 0); del buf598  # reuse
        # Source Nodes: [hidden_states_235, residual_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf582, buf599, arg371_1, arg372_1, arg373_1, buf603, 128, 1024, grid=grid(128), stream=stream0)
        del arg372_1
        del arg373_1
        buf604 = reinterpret_tensor(buf563, (128, 4096), (4096, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf603, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg374_1, (1024, 4096), (1, 1024), 0), out=buf604)
        del arg374_1
        buf605 = reinterpret_tensor(buf604, (1, 128, 4096), (524288, 4096, 1), 0); del buf604  # reuse
        # Source Nodes: [hidden_states_236], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf605, arg375_1, 524288, grid=grid(524288), stream=stream0)
        del arg375_1
        buf606 = reinterpret_tensor(buf603, (128, 1024), (1024, 1), 0); del buf603  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf605, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg376_1, (4096, 1024), (1, 4096), 0), out=buf606)
        del arg376_1
        buf607 = reinterpret_tensor(buf606, (1, 128, 1024), (131072, 1024, 1), 0); del buf606  # reuse
        buf611 = reinterpret_tensor(buf587, (1, 128, 1024), (131072, 1024, 1), 0); del buf587  # reuse
        # Source Nodes: [hidden_states_242, residual_44, residual_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf607, buf582, buf599, arg371_1, arg377_1, arg378_1, arg379_1, buf611, 128, 1024, grid=grid(128), stream=stream0)
        del arg371_1
        del arg377_1
        del arg378_1
        del arg379_1
        buf612 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf611, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg380_1, (1024, 1024), (1, 1024), 0), out=buf612)
        del arg380_1
        buf613 = reinterpret_tensor(buf582, (128, 1024), (1024, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf611, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg382_1, (1024, 1024), (1, 1024), 0), out=buf613)
        del arg382_1
        buf614 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_52], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf613, arg383_1, buf614, 131072, grid=grid(131072), stream=stream0)
        del arg383_1
        buf615 = reinterpret_tensor(buf613, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf613  # reuse
        # Source Nodes: [contiguous_80], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf612, arg381_1, buf615, 131072, grid=grid(131072), stream=stream0)
        del arg381_1
        buf616 = buf578; del buf578  # reuse
        # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf615, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf614, (16, 64, 128), (8192, 1, 64), 0), out=buf616)
        buf621 = buf573; del buf573  # reuse
        # Source Nodes: [attn_weights_69], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf616, buf621, 2048, 128, grid=grid(2048), stream=stream0)
        buf619 = reinterpret_tensor(buf615, (128, 1024), (1024, 1), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf611, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg384_1, (1024, 1024), (1, 1024), 0), out=buf619)
        del arg384_1
        buf620 = reinterpret_tensor(buf611, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf611  # reuse
        # Source Nodes: [value_states_52], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf619, arg385_1, buf620, 131072, grid=grid(131072), stream=stream0)
        del arg385_1
        buf622 = reinterpret_tensor(buf619, (16, 128, 64), (8192, 64, 1), 0); del buf619  # reuse
        # Source Nodes: [attn_output_130, attn_weights_69], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf621, reinterpret_tensor(buf620, (16, 128, 64), (8192, 64, 1), 0), out=buf622)
        buf623 = reinterpret_tensor(buf612, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf612  # reuse
        # Source Nodes: [attn_output_133], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf622, buf623, 131072, grid=grid(131072), stream=stream0)
        buf624 = reinterpret_tensor(buf622, (128, 1024), (1024, 1), 0); del buf622  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf623, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg386_1, (1024, 1024), (1, 1024), 0), out=buf624)
        del arg386_1
        buf628 = reinterpret_tensor(buf623, (1, 128, 1024), (131072, 1024, 1), 0); del buf623  # reuse
        # Source Nodes: [hidden_states_246, residual_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf607, buf624, arg387_1, arg388_1, arg389_1, buf628, 128, 1024, grid=grid(128), stream=stream0)
        del arg388_1
        del arg389_1
        buf629 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf628, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg390_1, (1024, 1024), (1, 1024), 0), out=buf629)
        del arg390_1
        buf630 = reinterpret_tensor(buf628, (128, 1024), (1024, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg392_1, (1024, 1024), (1, 1024), 0), out=buf630)
        del arg392_1
        buf631 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_54], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf630, arg393_1, buf631, 131072, grid=grid(131072), stream=stream0)
        del arg393_1
        buf632 = buf630; del buf630  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg394_1, (1024, 1024), (1, 1024), 0), out=buf632)
        del arg394_1
        buf633 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_54], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf632, arg395_1, buf633, 131072, grid=grid(131072), stream=stream0)
        del arg395_1
        buf634 = reinterpret_tensor(buf632, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf632  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf629, arg391_1, buf634, 131072, grid=grid(131072), stream=stream0)
        del arg391_1
        # Source Nodes: [], Original ATen: []
        buf635 = aten._scaled_dot_product_efficient_attention(buf634, reinterpret_tensor(buf631, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf633, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf636 = buf635[0]
        del buf635
        buf640 = reinterpret_tensor(buf636, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf636  # reuse
        # Source Nodes: [attn_output_138], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf640, 131072, grid=grid(131072), stream=stream0)
        buf641 = reinterpret_tensor(buf634, (128, 1024), (1024, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf640, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg396_1, (1024, 1024), (1, 1024), 0), out=buf641)
        del arg396_1
        buf642 = reinterpret_tensor(buf641, (1, 128, 1024), (131072, 1024, 1), 0); del buf641  # reuse
        buf646 = reinterpret_tensor(buf640, (1, 128, 1024), (131072, 1024, 1), 0); del buf640  # reuse
        # Source Nodes: [hidden_states_250, residual_46, residual_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf642, buf607, buf624, arg387_1, arg397_1, arg398_1, arg399_1, buf646, 128, 1024, grid=grid(128), stream=stream0)
        del arg387_1
        del arg397_1
        del arg398_1
        del arg399_1
        buf647 = reinterpret_tensor(buf605, (128, 4096), (4096, 1), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf646, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg400_1, (1024, 4096), (1, 1024), 0), out=buf647)
        del arg400_1
        buf648 = reinterpret_tensor(buf647, (1, 128, 4096), (524288, 4096, 1), 0); del buf647  # reuse
        # Source Nodes: [hidden_states_251], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf648, arg401_1, 524288, grid=grid(524288), stream=stream0)
        del arg401_1
        buf649 = reinterpret_tensor(buf646, (128, 1024), (1024, 1), 0); del buf646  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf648, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg402_1, (4096, 1024), (1, 4096), 0), out=buf649)
        del arg402_1
        buf653 = reinterpret_tensor(buf624, (1, 128, 1024), (131072, 1024, 1), 0); del buf624  # reuse
        # Source Nodes: [hidden_states_257, residual_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf642, buf649, arg403_1, arg404_1, arg405_1, buf653, 128, 1024, grid=grid(128), stream=stream0)
        del arg404_1
        del arg405_1
        buf654 = reinterpret_tensor(buf607, (128, 1024), (1024, 1), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf653, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg406_1, (1024, 1024), (1, 1024), 0), out=buf654)
        del arg406_1
        buf655 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf653, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg408_1, (1024, 1024), (1, 1024), 0), out=buf655)
        del arg408_1
        buf656 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_56], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf655, arg409_1, buf656, 131072, grid=grid(131072), stream=stream0)
        del arg409_1
        buf657 = reinterpret_tensor(buf655, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf655  # reuse
        # Source Nodes: [contiguous_86], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf654, arg407_1, buf657, 131072, grid=grid(131072), stream=stream0)
        del arg407_1
        buf658 = buf621; del buf621  # reuse
        # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf657, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf656, (16, 64, 128), (8192, 1, 64), 0), out=buf658)
        buf663 = buf616; del buf616  # reuse
        # Source Nodes: [attn_weights_75], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf658, buf663, 2048, 128, grid=grid(2048), stream=stream0)
        buf661 = reinterpret_tensor(buf657, (128, 1024), (1024, 1), 0); del buf657  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf653, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg410_1, (1024, 1024), (1, 1024), 0), out=buf661)
        del arg410_1
        buf662 = reinterpret_tensor(buf653, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf653  # reuse
        # Source Nodes: [value_states_56], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf661, arg411_1, buf662, 131072, grid=grid(131072), stream=stream0)
        del arg411_1
        buf664 = reinterpret_tensor(buf661, (16, 128, 64), (8192, 64, 1), 0); del buf661  # reuse
        # Source Nodes: [attn_output_140, attn_weights_75], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf663, reinterpret_tensor(buf662, (16, 128, 64), (8192, 64, 1), 0), out=buf664)
        buf665 = reinterpret_tensor(buf654, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf654  # reuse
        # Source Nodes: [attn_output_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf664, buf665, 131072, grid=grid(131072), stream=stream0)
        buf666 = reinterpret_tensor(buf664, (128, 1024), (1024, 1), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf665, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg412_1, (1024, 1024), (1, 1024), 0), out=buf666)
        del arg412_1
        buf667 = reinterpret_tensor(buf666, (1, 128, 1024), (131072, 1024, 1), 0); del buf666  # reuse
        buf671 = reinterpret_tensor(buf665, (1, 128, 1024), (131072, 1024, 1), 0); del buf665  # reuse
        # Source Nodes: [hidden_states_261, residual_48, residual_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf667, buf642, buf649, arg403_1, arg413_1, arg414_1, arg415_1, buf671, 128, 1024, grid=grid(128), stream=stream0)
        del arg403_1
        del arg413_1
        del arg414_1
        del arg415_1
        buf672 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf671, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg416_1, (1024, 1024), (1, 1024), 0), out=buf672)
        del arg416_1
        buf673 = reinterpret_tensor(buf671, (128, 1024), (1024, 1), 0); del buf671  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg418_1, (1024, 1024), (1, 1024), 0), out=buf673)
        del arg418_1
        buf674 = reinterpret_tensor(buf642, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf642  # reuse
        # Source Nodes: [key_states_58], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf673, arg419_1, buf674, 131072, grid=grid(131072), stream=stream0)
        del arg419_1
        buf675 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg420_1, (1024, 1024), (1, 1024), 0), out=buf675)
        del arg420_1
        buf676 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_58], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf675, arg421_1, buf676, 131072, grid=grid(131072), stream=stream0)
        del arg421_1
        buf677 = reinterpret_tensor(buf675, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf672, arg417_1, buf677, 131072, grid=grid(131072), stream=stream0)
        del arg417_1
        # Source Nodes: [], Original ATen: []
        buf678 = aten._scaled_dot_product_efficient_attention(buf677, reinterpret_tensor(buf674, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf676, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf679 = buf678[0]
        del buf678
        buf683 = reinterpret_tensor(buf679, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf679  # reuse
        # Source Nodes: [attn_output_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf683, 131072, grid=grid(131072), stream=stream0)
        buf684 = reinterpret_tensor(buf677, (128, 1024), (1024, 1), 0); del buf677  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf683, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg422_1, (1024, 1024), (1, 1024), 0), out=buf684)
        del arg422_1
        buf688 = reinterpret_tensor(buf683, (1, 128, 1024), (131072, 1024, 1), 0); del buf683  # reuse
        # Source Nodes: [hidden_states_265, residual_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf667, buf684, arg423_1, arg424_1, arg425_1, buf688, 128, 1024, grid=grid(128), stream=stream0)
        del arg424_1
        del arg425_1
        buf689 = reinterpret_tensor(buf648, (128, 4096), (4096, 1), 0); del buf648  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf688, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg426_1, (1024, 4096), (1, 1024), 0), out=buf689)
        del arg426_1
        buf690 = reinterpret_tensor(buf689, (1, 128, 4096), (524288, 4096, 1), 0); del buf689  # reuse
        # Source Nodes: [hidden_states_266], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf690, arg427_1, 524288, grid=grid(524288), stream=stream0)
        del arg427_1
        buf691 = reinterpret_tensor(buf688, (128, 1024), (1024, 1), 0); del buf688  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf690, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg428_1, (4096, 1024), (1, 4096), 0), out=buf691)
        del arg428_1
        buf692 = reinterpret_tensor(buf691, (1, 128, 1024), (131072, 1024, 1), 0); del buf691  # reuse
        buf696 = reinterpret_tensor(buf672, (1, 128, 1024), (131072, 1024, 1), 0); del buf672  # reuse
        # Source Nodes: [hidden_states_272, residual_50, residual_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf692, buf667, buf684, arg423_1, arg429_1, arg430_1, arg431_1, buf696, 128, 1024, grid=grid(128), stream=stream0)
        del arg423_1
        del arg429_1
        del arg430_1
        del arg431_1
        buf697 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf696, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg432_1, (1024, 1024), (1, 1024), 0), out=buf697)
        del arg432_1
        buf698 = reinterpret_tensor(buf667, (128, 1024), (1024, 1), 0); del buf667  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf696, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg434_1, (1024, 1024), (1, 1024), 0), out=buf698)
        del arg434_1
        buf699 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_60], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf698, arg435_1, buf699, 131072, grid=grid(131072), stream=stream0)
        del arg435_1
        buf700 = reinterpret_tensor(buf698, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf698  # reuse
        # Source Nodes: [contiguous_92], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf697, arg433_1, buf700, 131072, grid=grid(131072), stream=stream0)
        del arg433_1
        buf701 = buf663; del buf663  # reuse
        # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf700, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf699, (16, 64, 128), (8192, 1, 64), 0), out=buf701)
        buf706 = buf658; del buf658  # reuse
        # Source Nodes: [attn_weights_81], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf701, buf706, 2048, 128, grid=grid(2048), stream=stream0)
        buf704 = reinterpret_tensor(buf700, (128, 1024), (1024, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf696, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg436_1, (1024, 1024), (1, 1024), 0), out=buf704)
        del arg436_1
        buf705 = reinterpret_tensor(buf696, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf696  # reuse
        # Source Nodes: [value_states_60], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf704, arg437_1, buf705, 131072, grid=grid(131072), stream=stream0)
        del arg437_1
        buf707 = reinterpret_tensor(buf704, (16, 128, 64), (8192, 64, 1), 0); del buf704  # reuse
        # Source Nodes: [attn_output_150, attn_weights_81], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf706, reinterpret_tensor(buf705, (16, 128, 64), (8192, 64, 1), 0), out=buf707)
        buf708 = reinterpret_tensor(buf697, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf697  # reuse
        # Source Nodes: [attn_output_153], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf707, buf708, 131072, grid=grid(131072), stream=stream0)
        buf709 = reinterpret_tensor(buf707, (128, 1024), (1024, 1), 0); del buf707  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf708, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg438_1, (1024, 1024), (1, 1024), 0), out=buf709)
        del arg438_1
        buf713 = reinterpret_tensor(buf708, (1, 128, 1024), (131072, 1024, 1), 0); del buf708  # reuse
        # Source Nodes: [hidden_states_276, residual_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf692, buf709, arg439_1, arg440_1, arg441_1, buf713, 128, 1024, grid=grid(128), stream=stream0)
        del arg440_1
        del arg441_1
        buf714 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf713, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg442_1, (1024, 1024), (1, 1024), 0), out=buf714)
        del arg442_1
        buf715 = reinterpret_tensor(buf713, (128, 1024), (1024, 1), 0); del buf713  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg444_1, (1024, 1024), (1, 1024), 0), out=buf715)
        del arg444_1
        buf716 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_62], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf715, arg445_1, buf716, 131072, grid=grid(131072), stream=stream0)
        del arg445_1
        buf717 = buf715; del buf715  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg446_1, (1024, 1024), (1, 1024), 0), out=buf717)
        del arg446_1
        buf718 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_62], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf717, arg447_1, buf718, 131072, grid=grid(131072), stream=stream0)
        del arg447_1
        buf719 = reinterpret_tensor(buf717, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf714, arg443_1, buf719, 131072, grid=grid(131072), stream=stream0)
        del arg443_1
        # Source Nodes: [], Original ATen: []
        buf720 = aten._scaled_dot_product_efficient_attention(buf719, reinterpret_tensor(buf716, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf718, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf721 = buf720[0]
        del buf720
        buf725 = reinterpret_tensor(buf721, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf721  # reuse
        # Source Nodes: [attn_output_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf725, 131072, grid=grid(131072), stream=stream0)
        buf726 = reinterpret_tensor(buf719, (128, 1024), (1024, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf725, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg448_1, (1024, 1024), (1, 1024), 0), out=buf726)
        del arg448_1
        buf727 = reinterpret_tensor(buf726, (1, 128, 1024), (131072, 1024, 1), 0); del buf726  # reuse
        buf731 = reinterpret_tensor(buf725, (1, 128, 1024), (131072, 1024, 1), 0); del buf725  # reuse
        # Source Nodes: [hidden_states_280, residual_52, residual_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf727, buf692, buf709, arg439_1, arg449_1, arg450_1, arg451_1, buf731, 128, 1024, grid=grid(128), stream=stream0)
        del arg439_1
        del arg449_1
        del arg450_1
        del arg451_1
        buf732 = reinterpret_tensor(buf690, (128, 4096), (4096, 1), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf731, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg452_1, (1024, 4096), (1, 1024), 0), out=buf732)
        del arg452_1
        buf733 = reinterpret_tensor(buf732, (1, 128, 4096), (524288, 4096, 1), 0); del buf732  # reuse
        # Source Nodes: [hidden_states_281], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf733, arg453_1, 524288, grid=grid(524288), stream=stream0)
        del arg453_1
        buf734 = reinterpret_tensor(buf731, (128, 1024), (1024, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf733, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg454_1, (4096, 1024), (1, 4096), 0), out=buf734)
        del arg454_1
        buf738 = reinterpret_tensor(buf709, (1, 128, 1024), (131072, 1024, 1), 0); del buf709  # reuse
        # Source Nodes: [hidden_states_287, residual_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf727, buf734, arg455_1, arg456_1, arg457_1, buf738, 128, 1024, grid=grid(128), stream=stream0)
        del arg456_1
        del arg457_1
        buf739 = reinterpret_tensor(buf692, (128, 1024), (1024, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf738, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg458_1, (1024, 1024), (1, 1024), 0), out=buf739)
        del arg458_1
        buf740 = buf714; del buf714  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf738, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg460_1, (1024, 1024), (1, 1024), 0), out=buf740)
        del arg460_1
        buf741 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_64], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf740, arg461_1, buf741, 131072, grid=grid(131072), stream=stream0)
        del arg461_1
        buf742 = reinterpret_tensor(buf740, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf740  # reuse
        # Source Nodes: [contiguous_98], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf739, arg459_1, buf742, 131072, grid=grid(131072), stream=stream0)
        del arg459_1
        buf743 = buf706; del buf706  # reuse
        # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf742, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf741, (16, 64, 128), (8192, 1, 64), 0), out=buf743)
        buf748 = buf701; del buf701  # reuse
        # Source Nodes: [attn_weights_87], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf743, buf748, 2048, 128, grid=grid(2048), stream=stream0)
        buf746 = reinterpret_tensor(buf742, (128, 1024), (1024, 1), 0); del buf742  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf738, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg462_1, (1024, 1024), (1, 1024), 0), out=buf746)
        del arg462_1
        buf747 = reinterpret_tensor(buf738, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf738  # reuse
        # Source Nodes: [value_states_64], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf746, arg463_1, buf747, 131072, grid=grid(131072), stream=stream0)
        del arg463_1
        buf749 = reinterpret_tensor(buf746, (16, 128, 64), (8192, 64, 1), 0); del buf746  # reuse
        # Source Nodes: [attn_output_160, attn_weights_87], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf748, reinterpret_tensor(buf747, (16, 128, 64), (8192, 64, 1), 0), out=buf749)
        buf750 = reinterpret_tensor(buf739, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf739  # reuse
        # Source Nodes: [attn_output_163], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf749, buf750, 131072, grid=grid(131072), stream=stream0)
        buf751 = reinterpret_tensor(buf749, (128, 1024), (1024, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf750, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg464_1, (1024, 1024), (1, 1024), 0), out=buf751)
        del arg464_1
        buf752 = reinterpret_tensor(buf751, (1, 128, 1024), (131072, 1024, 1), 0); del buf751  # reuse
        buf756 = reinterpret_tensor(buf750, (1, 128, 1024), (131072, 1024, 1), 0); del buf750  # reuse
        # Source Nodes: [hidden_states_291, residual_54, residual_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf752, buf727, buf734, arg455_1, arg465_1, arg466_1, arg467_1, buf756, 128, 1024, grid=grid(128), stream=stream0)
        del arg455_1
        del arg465_1
        del arg466_1
        del arg467_1
        buf757 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf756, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg468_1, (1024, 1024), (1, 1024), 0), out=buf757)
        del arg468_1
        buf758 = reinterpret_tensor(buf756, (128, 1024), (1024, 1), 0); del buf756  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg470_1, (1024, 1024), (1, 1024), 0), out=buf758)
        del arg470_1
        buf759 = reinterpret_tensor(buf727, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf727  # reuse
        # Source Nodes: [key_states_66], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf758, arg471_1, buf759, 131072, grid=grid(131072), stream=stream0)
        del arg471_1
        buf760 = buf758; del buf758  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg472_1, (1024, 1024), (1, 1024), 0), out=buf760)
        del arg472_1
        buf761 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_66], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf760, arg473_1, buf761, 131072, grid=grid(131072), stream=stream0)
        del arg473_1
        buf762 = reinterpret_tensor(buf760, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf757, arg469_1, buf762, 131072, grid=grid(131072), stream=stream0)
        del arg469_1
        # Source Nodes: [], Original ATen: []
        buf763 = aten._scaled_dot_product_efficient_attention(buf762, reinterpret_tensor(buf759, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf761, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf764 = buf763[0]
        del buf763
        buf768 = reinterpret_tensor(buf764, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf764  # reuse
        # Source Nodes: [attn_output_168], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf768, 131072, grid=grid(131072), stream=stream0)
        buf769 = reinterpret_tensor(buf762, (128, 1024), (1024, 1), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf768, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg474_1, (1024, 1024), (1, 1024), 0), out=buf769)
        del arg474_1
        buf773 = reinterpret_tensor(buf768, (1, 128, 1024), (131072, 1024, 1), 0); del buf768  # reuse
        # Source Nodes: [hidden_states_295, residual_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf752, buf769, arg475_1, arg476_1, arg477_1, buf773, 128, 1024, grid=grid(128), stream=stream0)
        del arg476_1
        del arg477_1
        buf774 = reinterpret_tensor(buf733, (128, 4096), (4096, 1), 0); del buf733  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf773, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg478_1, (1024, 4096), (1, 1024), 0), out=buf774)
        del arg478_1
        buf775 = reinterpret_tensor(buf774, (1, 128, 4096), (524288, 4096, 1), 0); del buf774  # reuse
        # Source Nodes: [hidden_states_296], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf775, arg479_1, 524288, grid=grid(524288), stream=stream0)
        del arg479_1
        buf776 = reinterpret_tensor(buf773, (128, 1024), (1024, 1), 0); del buf773  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf775, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg480_1, (4096, 1024), (1, 4096), 0), out=buf776)
        del arg480_1
        buf777 = reinterpret_tensor(buf776, (1, 128, 1024), (131072, 1024, 1), 0); del buf776  # reuse
        buf781 = reinterpret_tensor(buf757, (1, 128, 1024), (131072, 1024, 1), 0); del buf757  # reuse
        # Source Nodes: [hidden_states_302, residual_56, residual_57], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf777, buf752, buf769, arg475_1, arg481_1, arg482_1, arg483_1, buf781, 128, 1024, grid=grid(128), stream=stream0)
        del arg475_1
        del arg481_1
        del arg482_1
        del arg483_1
        buf782 = buf769; del buf769  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf781, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg484_1, (1024, 1024), (1, 1024), 0), out=buf782)
        del arg484_1
        buf783 = reinterpret_tensor(buf752, (128, 1024), (1024, 1), 0); del buf752  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf781, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg486_1, (1024, 1024), (1, 1024), 0), out=buf783)
        del arg486_1
        buf784 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_68], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf783, arg487_1, buf784, 131072, grid=grid(131072), stream=stream0)
        del arg487_1
        buf785 = reinterpret_tensor(buf783, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf783  # reuse
        # Source Nodes: [contiguous_104], Original ATen: [aten.clone]
        triton_poi_fused_2.run(buf782, arg485_1, buf785, 131072, grid=grid(131072), stream=stream0)
        del arg485_1
        buf786 = buf748; del buf748  # reuse
        # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf785, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf784, (16, 64, 128), (8192, 1, 64), 0), out=buf786)
        buf791 = buf743; del buf743  # reuse
        # Source Nodes: [attn_weights_93], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf786, buf791, 2048, 128, grid=grid(2048), stream=stream0)
        del buf786
        buf789 = reinterpret_tensor(buf785, (128, 1024), (1024, 1), 0); del buf785  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf781, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg488_1, (1024, 1024), (1, 1024), 0), out=buf789)
        del arg488_1
        buf790 = reinterpret_tensor(buf781, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf781  # reuse
        # Source Nodes: [value_states_68], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf789, arg489_1, buf790, 131072, grid=grid(131072), stream=stream0)
        del arg489_1
        buf792 = reinterpret_tensor(buf789, (16, 128, 64), (8192, 64, 1), 0); del buf789  # reuse
        # Source Nodes: [attn_output_170, attn_weights_93], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf791, reinterpret_tensor(buf790, (16, 128, 64), (8192, 64, 1), 0), out=buf792)
        del buf791
        buf793 = reinterpret_tensor(buf782, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf782  # reuse
        # Source Nodes: [attn_output_173], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf792, buf793, 131072, grid=grid(131072), stream=stream0)
        buf794 = reinterpret_tensor(buf792, (128, 1024), (1024, 1), 0); del buf792  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf793, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg490_1, (1024, 1024), (1, 1024), 0), out=buf794)
        del arg490_1
        buf798 = reinterpret_tensor(buf793, (1, 128, 1024), (131072, 1024, 1), 0); del buf793  # reuse
        # Source Nodes: [hidden_states_306, residual_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf777, buf794, arg491_1, arg492_1, arg493_1, buf798, 128, 1024, grid=grid(128), stream=stream0)
        del arg492_1
        del arg493_1
        buf799 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf798, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg494_1, (1024, 1024), (1, 1024), 0), out=buf799)
        del arg494_1
        buf800 = reinterpret_tensor(buf798, (128, 1024), (1024, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg496_1, (1024, 1024), (1, 1024), 0), out=buf800)
        del arg496_1
        buf801 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_70], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf800, arg497_1, buf801, 131072, grid=grid(131072), stream=stream0)
        del arg497_1
        buf802 = buf800; del buf800  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg498_1, (1024, 1024), (1, 1024), 0), out=buf802)
        del arg498_1
        buf803 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_70], Original ATen: [aten.clone]
        triton_poi_fused_3.run(buf802, arg499_1, buf803, 131072, grid=grid(131072), stream=stream0)
        del arg499_1
        buf804 = reinterpret_tensor(buf802, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf799, arg495_1, buf804, 131072, grid=grid(131072), stream=stream0)
        del arg495_1
        del buf799
        # Source Nodes: [], Original ATen: []
        buf805 = aten._scaled_dot_product_efficient_attention(buf804, reinterpret_tensor(buf801, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), reinterpret_tensor(buf803, (1, 16, 128, 64), (131072, 8192, 64, 1), 0), None, True, scale=1.0)
        buf806 = buf805[0]
        del buf805
        buf810 = reinterpret_tensor(buf806, (1, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf806  # reuse
        # Source Nodes: [attn_output_178], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf810, 131072, grid=grid(131072), stream=stream0)
        buf811 = reinterpret_tensor(buf804, (128, 1024), (1024, 1), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf810, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg500_1, (1024, 1024), (1, 1024), 0), out=buf811)
        del arg500_1
        buf812 = reinterpret_tensor(buf811, (1, 128, 1024), (131072, 1024, 1), 0); del buf811  # reuse
        buf816 = reinterpret_tensor(buf810, (1, 128, 1024), (131072, 1024, 1), 0); del buf810  # reuse
        # Source Nodes: [hidden_states_310, residual_58, residual_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf812, buf777, buf794, arg491_1, arg501_1, arg502_1, arg503_1, buf816, 128, 1024, grid=grid(128), stream=stream0)
        del arg491_1
        del arg501_1
        del arg502_1
        del arg503_1
        del buf777
        buf817 = reinterpret_tensor(buf775, (128, 4096), (4096, 1), 0); del buf775  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf816, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg504_1, (1024, 4096), (1, 1024), 0), out=buf817)
        del arg504_1
        buf818 = reinterpret_tensor(buf817, (1, 128, 4096), (524288, 4096, 1), 0); del buf817  # reuse
        # Source Nodes: [hidden_states_311], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf818, arg505_1, 524288, grid=grid(524288), stream=stream0)
        del arg505_1
        buf819 = reinterpret_tensor(buf816, (128, 1024), (1024, 1), 0); del buf816  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf818, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg506_1, (4096, 1024), (1, 4096), 0), out=buf819)
        del arg506_1
        del buf818
        buf823 = reinterpret_tensor(buf794, (1, 128, 1024), (131072, 1024, 1), 0); del buf794  # reuse
        # Source Nodes: [hidden_states_316, hidden_states_317], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf812, buf819, arg507_1, arg508_1, arg509_1, buf823, 128, 1024, grid=grid(128), stream=stream0)
        del arg507_1
        del arg508_1
        del arg509_1
        del buf812
        del buf819
        buf824 = empty((128, 128112), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf823, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg510_1, (1024, 128112), (1, 1024), 0), out=buf824)
        del arg510_1
        del buf823
        buf825 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_11.run(buf824, buf825, 512, 32028, grid=grid(512), stream=stream0)
        buf826 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_12.run(buf825, buf826, 128, 4, grid=grid(128), stream=stream0)
        buf827 = buf825; del buf825  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_13.run(buf824, buf826, buf827, 512, 32028, grid=grid(512), stream=stream0)
        buf828 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_14.run(buf827, buf828, 128, 4, grid=grid(128), stream=stream0)
        del buf827
        buf829 = empty((), device='cuda', dtype=torch.float32)
        buf831 = buf829; del buf829  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_15.run(buf831, arg513_1, buf824, buf826, buf828, 1, 128, grid=grid(1), stream=stream0)
        del arg513_1
        return (buf831, reinterpret_tensor(buf824, (1, 128, 128112), (16398336, 128112, 1), 0), buf315, buf321, buf334, buf336, buf359, buf365, buf376, buf378, buf401, buf407, buf419, buf421, buf444, buf450, buf461, buf463, buf486, buf492, buf504, buf506, buf529, buf535, buf546, buf548, buf571, buf577, buf589, buf591, buf614, buf620, buf631, buf633, buf656, buf662, buf674, buf676, buf699, buf705, buf716, buf718, buf741, buf747, buf759, buf761, buf784, buf790, buf801, buf803, buf332, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg514_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg515_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('M2M100ForConditionalGeneration', benchmark_compiled_module)
