
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqfzeow2rfwzhypvws4memq4rvzkpu6z4mgzxfxm44f4s3qkzre.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_6
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32014336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/oj/cojib5cjtva2qww365umgv3zoxwow4fqvapujgxd7ejhr7zbxrys.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_6
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
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
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/caioqmrbp6fbwav6j74ksr3ngjkpujgtsdixj4lyenkkfnenybk5.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 62528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 4)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (62528*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqeqd4obdtct7s6hhvg7aywslt5qz5v34w4kedmovsuq5iqdlhs.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zt/cztj3gofhbkmguewej42jcpp56673yrevsrwesxf63g6jycy5s3j.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32014336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 250112)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr4 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp13 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp9 = tmp6 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tl.exp(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 - tmp16
    tmp18 = tmp0 + tmp17
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7op2tvvglfzb5viye3illjveboufb622razwcdcgttbqqns2bi.py
# Source Nodes: [hidden_states_230], Original ATen: [aten.mul, aten.native_dropout_backward, aten.sum]
# hidden_states_230 => mul_169
triton_per_fused_mul_native_dropout_backward_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_dropout_backward_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckmgd3asv3pybbfnxlhtqzsyzfwkdjpsidqqcwijqhqsts3ppd2d.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tmp7 * tmp14
    tmp16 = -0.5
    tmp17 = tmp13 * tmp16
    tmp18 = tmp14 * tmp14
    tmp19 = tmp18 * tmp14
    tmp20 = tmp17 * tmp19
    tmp21 = 512.0
    tmp22 = tmp20 / tmp21
    tmp23 = 2.0
    tmp24 = tmp8 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp15 + tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp3
    tmp30 = tmp26 * tmp29
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cglwoyekm2deulu6b3n6w5jae7emvmbcgzmh2ko4q6chozpkizne.py
# Source Nodes: [add_118, hidden_gelu_15, mul_164], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
# add_118 => add_143
# hidden_gelu_15 => mul_167
# mul_164 => mul_164
triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp14 = tl.load(in_ptr4 + (x0), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 * tmp11
    tmp13 = tmp5 * tmp12
    tmp15 = tmp5 * tmp14
    tmp16 = tmp15 * tmp8
    tmp17 = tmp9 * tmp9
    tmp18 = tmp10 - tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = 0.7978845608028654
    tmp21 = tmp19 * tmp20
    tmp22 = 0.044715
    tmp23 = tmp21 * tmp22
    tmp24 = tmp6 * tmp6
    tmp25 = 3.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = tmp21 + tmp27
    tmp29 = tmp15 * tmp11
    tmp30 = tmp29 * tmp7
    tmp31 = tmp28 + tmp30
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(in_out_ptr0 + (x0), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpd5akfpecobfurl3jbiv5jyoi4hywghyhevjmnngyocsjssxyp6.py
# Source Nodes: [hidden_states_223], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_223 => mul_162
triton_per_fused_add_mul_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjvxtzzy2wwqpsodf4vbo2dnnixvihn4g4wyqmcd7ux6gssme7j.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp4 * tmp12
    tmp14 = tmp11 + tmp13
    tmp15 = -0.5
    tmp16 = tmp10 * tmp15
    tmp17 = tmp12 * tmp12
    tmp18 = tmp17 * tmp12
    tmp19 = tmp16 * tmp18
    tmp20 = 512.0
    tmp21 = tmp19 / tmp20
    tmp22 = 2.0
    tmp23 = tmp5 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp14 + tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsdcb3ywfvlzyoeuxbdy4yxub2a6id2y6ayisi7w6quszdemszj.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6es7d6vhkmj6ig5hzxnbsfogubz6azvqntiqcox6kf4pstyuexw.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]

triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11', 'mutated_arg_names': ['out_ptr2']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/chabfas57cnhiws6w4e77f43jjaksnnpvsvhlhzmh7eqy27t5jaa.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_12', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqo2uzxg7jereqi7tdrlqkljuldsfyan4vwk6vmx2qcsb3ofe6k.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (384*x1)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogr2fktgpovjrzagbgxdvtedqdatwqfn642g7bod346l3hjrbhn.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (128*x2) + (8192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (64*y1) + (384*y0)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbumpvrfbeb4sl37mv4vufg6stfztgcpmew72p5e7fsjzd6fqsq4.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyik3flc5q6rbnmrqrhvvmnqnltstqtfqa4xze4hfzuojogruiam.py
# Source Nodes: [hidden_states_219], Original ATen: [aten.mul, aten.sum]
# hidden_states_219 => mul_160
triton_per_fused_mul_sum_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bwlhgzan6jzl7q57p6qzh4ookdc34pmq4ucupllqfaze7gevlq.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp2 * tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = -0.5
    tmp14 = tmp8 * tmp13
    tmp15 = tmp10 * tmp10
    tmp16 = tmp15 * tmp10
    tmp17 = tmp14 * tmp16
    tmp18 = 512.0
    tmp19 = tmp17 / tmp18
    tmp20 = 2.0
    tmp21 = tmp3 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp12 + tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 1.1111111111111112
    tmp27 = tmp25 * tmp26
    tmp28 = tmp23 * tmp27
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp28, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pp/cppd4x4cnfz5hp5wiyzpykdgky3a2sh4qbomhx4dr56y3ypvhotp.py
# Source Nodes: [hidden_states_214], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_214 => mul_158
triton_per_fused_add_mul_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5grpyviastoq33cn2cykhgdzbrnqjsaarzq2q2k7gkwc2jbhvt.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = -0.5
    tmp18 = tmp12 * tmp17
    tmp19 = tmp14 * tmp14
    tmp20 = tmp19 * tmp14
    tmp21 = tmp18 * tmp20
    tmp22 = 512.0
    tmp23 = tmp21 / tmp22
    tmp24 = 2.0
    tmp25 = tmp7 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp16 + tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 1.1111111111111112
    tmp31 = tmp29 * tmp30
    tmp32 = tmp27 * tmp31
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp32, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctobo4mpwuvti3fuwlud2uaryeeffdxxuyoyht6tvkaysbkszddy.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*i1', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*i1', 22: '*fp32', 23: '*fp32', 24: 'i32', 25: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(24, 25))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp38 = tl.load(in_ptr17 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr18 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp46 = tl.load(in_ptr19 + (x0), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 1.1111111111111112
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 * tmp40
    tmp42 = tl.broadcast_to(tmp41, [RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = triton_helpers.promote_to_tensor(tl.sum(tmp44, 0))
    tmp47 = tmp39 * tmp46
    tmp48 = -0.5
    tmp49 = tmp45 * tmp48
    tmp50 = tmp46 * tmp46
    tmp51 = tmp50 * tmp46
    tmp52 = tmp49 * tmp51
    tmp53 = 512.0
    tmp54 = tmp52 / tmp53
    tmp55 = 2.0
    tmp56 = tmp40 * tmp55
    tmp57 = tmp54 * tmp56
    tmp58 = tmp47 + tmp57
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 * tmp35
    tmp62 = tmp58 * tmp61
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp58, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp62, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jvnjorbnvgmtojivhz4xvisvqdr6cuicn7r5kjuzmc7pbboqbt.py
# Source Nodes: [loss], Original ATen: [aten.as_strided_scatter, aten.embedding_dense_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_21', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tl.load(in_ptr2 + (x0), None)
    tmp4 = tl.load(in_ptr3 + (x0), None)
    tmp6 = tl.load(in_ptr4 + (x0), None)
    tmp8 = tl.load(in_ptr5 + (x0), None)
    tmp10 = tl.load(in_ptr6 + (x0), None)
    tmp12 = tl.load(in_ptr7 + (x0), None)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13 + tmp0
    tmp15 = tl.full([1], False, tl.int1)
    tmp16 = 0.0
    tmp17 = tl.where(tmp15, tmp16, tmp14)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4w7do4jjxiee66lrl2zmwsgbsp3mtupfvlx3bikl3xxb47msr6r.py
# Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_poi_fused_embedding_dense_backward_nll_loss_forward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dvormng3f5mgie5ktvdwyrlukaiab3g3oe7fccg7wqm36zwmc6.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum]
# loss => full_default_7
triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*i64', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel):
    xnumel = 128
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = -0.5
    tmp18 = tmp12 * tmp17
    tmp19 = tmp14 * tmp14
    tmp20 = tmp19 * tmp14
    tmp21 = tmp18 * tmp20
    tmp22 = 512.0
    tmp23 = tmp21 / tmp22
    tmp24 = 2.0
    tmp25 = tmp7 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp16 + tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 1.1111111111111112
    tmp31 = tmp29 * tmp30
    tmp32 = tmp27 * tmp31
    tmp34 = tl.full([1], -1, tl.int64)
    tmp35 = tmp33 == tmp34
    tmp36 = 0.0
    tmp37 = tl.where(tmp35, tmp36, tmp32)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cixjda7ptlwhbqd35y3l7w4ayyrlgdin2dcoq7qajdcqlmvpkbvn.py
# Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_poi_fused_embedding_dense_backward_nll_loss_forward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128057344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5pe25lk4jpitot4f72obdmzjahozuqwchphzjayow3oi2vysuu.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128057344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_192, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, mm_4, tanh, mm_5, getitem_7, view_25, getitem_9, add_10, rsqrt_2, view_27, getitem_11, view_45, getitem_13, add_13, rsqrt_3, view_47, mm_11, tanh_1, mm_12, getitem_15, view_51, getitem_17, add_17, rsqrt_4, view_53, getitem_19, view_71, getitem_21, add_20, rsqrt_5, view_73, mm_18, tanh_2, mm_19, getitem_23, view_77, getitem_25, add_24, rsqrt_6, view_79, getitem_27, view_97, getitem_29, add_27, rsqrt_7, view_99, mm_25, tanh_3, mm_26, getitem_31, view_103, getitem_33, add_31, rsqrt_8, view_105, getitem_35, view_123, getitem_37, add_34, rsqrt_9, view_125, mm_32, tanh_4, mm_33, getitem_39, view_129, getitem_41, add_38, rsqrt_10, view_131, getitem_43, view_149, getitem_45, add_41, rsqrt_11, view_151, mm_39, tanh_5, mm_40, getitem_47, view_155, getitem_49, add_45, rsqrt_12, view_157, getitem_51, view_175, getitem_53, add_48, rsqrt_13, view_177, mm_46, tanh_6, mm_47, getitem_55, view_181, getitem_57, add_52, rsqrt_14, view_183, getitem_59, view_201, getitem_61, add_55, rsqrt_15, view_203, mm_53, tanh_7, mm_54, getitem_63, view_207, getitem_65, add_59, rsqrt_16, getitem_67, view_209, getitem_68, getitem_69, rsqrt_17, view_210, add_63, getitem_71, view_228, getitem_73, add_66, rsqrt_18, view_230, view_233, getitem_75, view_248, getitem_77, add_70, rsqrt_19, view_250, mm_64, tanh_8, mm_65, getitem_79, view_254, getitem_81, add_74, rsqrt_20, view_256, getitem_83, view_274, getitem_85, add_77, rsqrt_21, view_276, getitem_87, view_294, getitem_89, add_80, rsqrt_22, view_296, mm_75, tanh_9, mm_76, getitem_91, view_300, getitem_93, add_84, rsqrt_23, view_302, getitem_95, view_320, getitem_97, add_87, rsqrt_24, view_322, getitem_99, view_340, getitem_101, add_90, rsqrt_25, view_342, mm_86, tanh_10, mm_87, getitem_103, view_346, getitem_105, add_94, rsqrt_26, view_348, getitem_107, view_366, getitem_109, add_97, rsqrt_27, view_368, getitem_111, view_386, getitem_113, add_100, rsqrt_28, view_388, mm_97, tanh_11, mm_98, getitem_115, view_392, getitem_117, add_104, rsqrt_29, view_394, getitem_119, view_412, getitem_121, add_107, rsqrt_30, view_414, getitem_123, view_432, getitem_125, add_110, rsqrt_31, view_434, mm_108, tanh_12, mm_109, getitem_127, view_438, getitem_129, add_114, rsqrt_32, view_440, getitem_131, view_458, getitem_133, add_117, rsqrt_33, view_460, getitem_135, view_478, getitem_137, add_120, rsqrt_34, view_480, mm_119, tanh_13, mm_120, getitem_139, view_484, getitem_141, add_124, rsqrt_35, view_486, getitem_143, view_504, getitem_145, add_127, rsqrt_36, view_506, getitem_147, view_524, getitem_149, add_130, rsqrt_37, view_526, mm_130, tanh_14, mm_131, getitem_151, view_530, getitem_153, add_134, rsqrt_38, view_532, getitem_155, view_550, getitem_157, add_137, rsqrt_39, view_552, getitem_159, view_570, getitem_161, add_140, rsqrt_40, view_572, mm_141, tanh_15, mm_142, getitem_163, view_576, getitem_165, add_144, rsqrt_41, getitem_167, view_578, sub_30, convert_element_type_7, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, alias_87, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, alias_89, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, alias_93, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, alias_95, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, alias_99, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, alias_101, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, alias_105, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, alias_107, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, alias_111, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, alias_113, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, alias_117, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, alias_119, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, alias_123, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, alias_125, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, alias_129, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, alias_131, permute_750, permute_751, permute_756, permute_761, permute_766, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, alias_136, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, alias_140, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, alias_144, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, alias_148, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, alias_152, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, alias_156, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, alias_160, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, alias_164, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35 = args
    args.clear()
    assert_size_stride(primals_1, (512, ), (1, ))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (512, ), (1, ))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, ), (1, ))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (512, ), (1, ))
    assert_size_stride(primals_13, (512, ), (1, ))
    assert_size_stride(primals_14, (512, ), (1, ))
    assert_size_stride(primals_15, (512, ), (1, ))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_192, (1, 128), (128, 1))
    assert_size_stride(view, (1, 128), (128, 1))
    assert_size_stride(getitem, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(getitem_1, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_1, (128, 512), (512, 1))
    assert_size_stride(add_3, (128, 128), (128, 1))
    assert_size_stride(getitem_3, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_19, (128, 384), (384, 1))
    assert_size_stride(getitem_5, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_6, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_1, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_21, (128, 512), (512, 1))
    assert_size_stride(mm_4, (128, 1024), (1024, 1))
    assert_size_stride(tanh, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_5, (128, 1024), (1024, 1))
    assert_size_stride(getitem_7, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_25, (128, 1024), (1024, 1))
    assert_size_stride(getitem_9, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_10, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_2, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_27, (128, 512), (512, 1))
    assert_size_stride(getitem_11, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_45, (128, 384), (384, 1))
    assert_size_stride(getitem_13, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_13, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_3, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_47, (128, 512), (512, 1))
    assert_size_stride(mm_11, (128, 1024), (1024, 1))
    assert_size_stride(tanh_1, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_12, (128, 1024), (1024, 1))
    assert_size_stride(getitem_15, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_51, (128, 1024), (1024, 1))
    assert_size_stride(getitem_17, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_17, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_4, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_53, (128, 512), (512, 1))
    assert_size_stride(getitem_19, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_71, (128, 384), (384, 1))
    assert_size_stride(getitem_21, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_20, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_5, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_73, (128, 512), (512, 1))
    assert_size_stride(mm_18, (128, 1024), (1024, 1))
    assert_size_stride(tanh_2, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_19, (128, 1024), (1024, 1))
    assert_size_stride(getitem_23, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_77, (128, 1024), (1024, 1))
    assert_size_stride(getitem_25, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_24, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_6, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_79, (128, 512), (512, 1))
    assert_size_stride(getitem_27, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_97, (128, 384), (384, 1))
    assert_size_stride(getitem_29, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_27, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_7, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_99, (128, 512), (512, 1))
    assert_size_stride(mm_25, (128, 1024), (1024, 1))
    assert_size_stride(tanh_3, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_26, (128, 1024), (1024, 1))
    assert_size_stride(getitem_31, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_103, (128, 1024), (1024, 1))
    assert_size_stride(getitem_33, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_31, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_8, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_105, (128, 512), (512, 1))
    assert_size_stride(getitem_35, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_123, (128, 384), (384, 1))
    assert_size_stride(getitem_37, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_34, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_9, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_125, (128, 512), (512, 1))
    assert_size_stride(mm_32, (128, 1024), (1024, 1))
    assert_size_stride(tanh_4, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_33, (128, 1024), (1024, 1))
    assert_size_stride(getitem_39, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_129, (128, 1024), (1024, 1))
    assert_size_stride(getitem_41, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_38, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_10, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_131, (128, 512), (512, 1))
    assert_size_stride(getitem_43, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_149, (128, 384), (384, 1))
    assert_size_stride(getitem_45, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_41, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_11, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_151, (128, 512), (512, 1))
    assert_size_stride(mm_39, (128, 1024), (1024, 1))
    assert_size_stride(tanh_5, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_40, (128, 1024), (1024, 1))
    assert_size_stride(getitem_47, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_155, (128, 1024), (1024, 1))
    assert_size_stride(getitem_49, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_45, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_12, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_157, (128, 512), (512, 1))
    assert_size_stride(getitem_51, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_175, (128, 384), (384, 1))
    assert_size_stride(getitem_53, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_48, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_13, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_177, (128, 512), (512, 1))
    assert_size_stride(mm_46, (128, 1024), (1024, 1))
    assert_size_stride(tanh_6, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_47, (128, 1024), (1024, 1))
    assert_size_stride(getitem_55, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_181, (128, 1024), (1024, 1))
    assert_size_stride(getitem_57, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_52, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_14, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_183, (128, 512), (512, 1))
    assert_size_stride(getitem_59, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_201, (128, 384), (384, 1))
    assert_size_stride(getitem_61, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_55, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_15, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_203, (128, 512), (512, 1))
    assert_size_stride(mm_53, (128, 1024), (1024, 1))
    assert_size_stride(tanh_7, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_54, (128, 1024), (1024, 1))
    assert_size_stride(getitem_63, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_207, (128, 1024), (1024, 1))
    assert_size_stride(getitem_65, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_59, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_16, (1, 128, 1), (128, 1, 1))
    assert_size_stride(getitem_67, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_209, (1, 128), (128, 1))
    assert_size_stride(getitem_68, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(getitem_69, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_17, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_210, (128, 512), (512, 1))
    assert_size_stride(add_63, (128, 128), (128, 1))
    assert_size_stride(getitem_71, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_228, (128, 384), (384, 1))
    assert_size_stride(getitem_73, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_66, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_18, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_230, (128, 512), (512, 1))
    assert_size_stride(view_233, (128, 512), (512, 1))
    assert_size_stride(getitem_75, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_248, (128, 384), (384, 1))
    assert_size_stride(getitem_77, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_70, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_19, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_250, (128, 512), (512, 1))
    assert_size_stride(mm_64, (128, 1024), (1024, 1))
    assert_size_stride(tanh_8, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_65, (128, 1024), (1024, 1))
    assert_size_stride(getitem_79, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_254, (128, 1024), (1024, 1))
    assert_size_stride(getitem_81, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_74, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_20, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_256, (128, 512), (512, 1))
    assert_size_stride(getitem_83, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_274, (128, 384), (384, 1))
    assert_size_stride(getitem_85, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_77, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_21, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_276, (128, 512), (512, 1))
    assert_size_stride(getitem_87, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_294, (128, 384), (384, 1))
    assert_size_stride(getitem_89, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_80, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_22, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_296, (128, 512), (512, 1))
    assert_size_stride(mm_75, (128, 1024), (1024, 1))
    assert_size_stride(tanh_9, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_76, (128, 1024), (1024, 1))
    assert_size_stride(getitem_91, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_300, (128, 1024), (1024, 1))
    assert_size_stride(getitem_93, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_84, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_23, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_302, (128, 512), (512, 1))
    assert_size_stride(getitem_95, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_320, (128, 384), (384, 1))
    assert_size_stride(getitem_97, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_87, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_24, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_322, (128, 512), (512, 1))
    assert_size_stride(getitem_99, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_340, (128, 384), (384, 1))
    assert_size_stride(getitem_101, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_90, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_25, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_342, (128, 512), (512, 1))
    assert_size_stride(mm_86, (128, 1024), (1024, 1))
    assert_size_stride(tanh_10, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_87, (128, 1024), (1024, 1))
    assert_size_stride(getitem_103, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_346, (128, 1024), (1024, 1))
    assert_size_stride(getitem_105, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_94, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_26, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_348, (128, 512), (512, 1))
    assert_size_stride(getitem_107, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_366, (128, 384), (384, 1))
    assert_size_stride(getitem_109, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_97, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_368, (128, 512), (512, 1))
    assert_size_stride(getitem_111, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_386, (128, 384), (384, 1))
    assert_size_stride(getitem_113, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_100, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_388, (128, 512), (512, 1))
    assert_size_stride(mm_97, (128, 1024), (1024, 1))
    assert_size_stride(tanh_11, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_98, (128, 1024), (1024, 1))
    assert_size_stride(getitem_115, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_392, (128, 1024), (1024, 1))
    assert_size_stride(getitem_117, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_104, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_29, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_394, (128, 512), (512, 1))
    assert_size_stride(getitem_119, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_412, (128, 384), (384, 1))
    assert_size_stride(getitem_121, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_107, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_414, (128, 512), (512, 1))
    assert_size_stride(getitem_123, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_432, (128, 384), (384, 1))
    assert_size_stride(getitem_125, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_110, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_434, (128, 512), (512, 1))
    assert_size_stride(mm_108, (128, 1024), (1024, 1))
    assert_size_stride(tanh_12, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_109, (128, 1024), (1024, 1))
    assert_size_stride(getitem_127, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_438, (128, 1024), (1024, 1))
    assert_size_stride(getitem_129, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_114, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_32, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_440, (128, 512), (512, 1))
    assert_size_stride(getitem_131, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_458, (128, 384), (384, 1))
    assert_size_stride(getitem_133, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_117, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_460, (128, 512), (512, 1))
    assert_size_stride(getitem_135, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_478, (128, 384), (384, 1))
    assert_size_stride(getitem_137, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_120, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_34, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_480, (128, 512), (512, 1))
    assert_size_stride(mm_119, (128, 1024), (1024, 1))
    assert_size_stride(tanh_13, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_120, (128, 1024), (1024, 1))
    assert_size_stride(getitem_139, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_484, (128, 1024), (1024, 1))
    assert_size_stride(getitem_141, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_124, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_35, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_486, (128, 512), (512, 1))
    assert_size_stride(getitem_143, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_504, (128, 384), (384, 1))
    assert_size_stride(getitem_145, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_127, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_36, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_506, (128, 512), (512, 1))
    assert_size_stride(getitem_147, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_524, (128, 384), (384, 1))
    assert_size_stride(getitem_149, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_130, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_37, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_526, (128, 512), (512, 1))
    assert_size_stride(mm_130, (128, 1024), (1024, 1))
    assert_size_stride(tanh_14, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_131, (128, 1024), (1024, 1))
    assert_size_stride(getitem_151, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_530, (128, 1024), (1024, 1))
    assert_size_stride(getitem_153, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_134, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_38, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_532, (128, 512), (512, 1))
    assert_size_stride(getitem_155, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_550, (128, 384), (384, 1))
    assert_size_stride(getitem_157, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_137, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_39, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_552, (128, 512), (512, 1))
    assert_size_stride(getitem_159, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_570, (128, 384), (384, 1))
    assert_size_stride(getitem_161, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_140, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_40, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_572, (128, 512), (512, 1))
    assert_size_stride(mm_141, (128, 1024), (1024, 1))
    assert_size_stride(tanh_15, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_142, (128, 1024), (1024, 1))
    assert_size_stride(getitem_163, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_576, (128, 1024), (1024, 1))
    assert_size_stride(getitem_165, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_144, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_41, (1, 128, 1), (128, 1, 1))
    assert_size_stride(getitem_167, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_578, (128, 512), (512, 1))
    assert_size_stride(sub_30, (128, 250112), (250112, 1))
    assert_size_stride(convert_element_type_7, (), ())
    assert_size_stride(permute_269, (250112, 512), (512, 1))
    assert_size_stride(permute_273, (512, 1024), (1024, 1))
    assert_size_stride(permute_277, (1024, 512), (512, 1))
    assert_size_stride(permute_281, (1024, 512), (512, 1))
    assert_size_stride(permute_285, (512, 384), (384, 1))
    assert_size_stride(permute_288, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_289, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_87, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_290, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_291, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_296, (384, 512), (512, 1))
    assert_size_stride(permute_301, (384, 512), (512, 1))
    assert_size_stride(permute_306, (384, 512), (512, 1))
    assert_size_stride(permute_310, (512, 384), (384, 1))
    assert_size_stride(permute_313, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_314, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_89, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_315, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_316, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_321, (384, 512), (512, 1))
    assert_size_stride(permute_326, (384, 512), (512, 1))
    assert_size_stride(permute_331, (384, 512), (512, 1))
    assert_size_stride(permute_335, (512, 1024), (1024, 1))
    assert_size_stride(permute_339, (1024, 512), (512, 1))
    assert_size_stride(permute_343, (1024, 512), (512, 1))
    assert_size_stride(permute_347, (512, 384), (384, 1))
    assert_size_stride(permute_350, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_351, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_93, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_352, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_353, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_358, (384, 512), (512, 1))
    assert_size_stride(permute_363, (384, 512), (512, 1))
    assert_size_stride(permute_368, (384, 512), (512, 1))
    assert_size_stride(permute_372, (512, 384), (384, 1))
    assert_size_stride(permute_375, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_376, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_95, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_377, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_378, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_383, (384, 512), (512, 1))
    assert_size_stride(permute_388, (384, 512), (512, 1))
    assert_size_stride(permute_393, (384, 512), (512, 1))
    assert_size_stride(permute_397, (512, 1024), (1024, 1))
    assert_size_stride(permute_401, (1024, 512), (512, 1))
    assert_size_stride(permute_405, (1024, 512), (512, 1))
    assert_size_stride(permute_409, (512, 384), (384, 1))
    assert_size_stride(permute_412, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_413, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_99, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_414, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_415, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_420, (384, 512), (512, 1))
    assert_size_stride(permute_425, (384, 512), (512, 1))
    assert_size_stride(permute_430, (384, 512), (512, 1))
    assert_size_stride(permute_434, (512, 384), (384, 1))
    assert_size_stride(permute_437, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_438, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_101, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_439, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_440, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_445, (384, 512), (512, 1))
    assert_size_stride(permute_450, (384, 512), (512, 1))
    assert_size_stride(permute_455, (384, 512), (512, 1))
    assert_size_stride(permute_459, (512, 1024), (1024, 1))
    assert_size_stride(permute_463, (1024, 512), (512, 1))
    assert_size_stride(permute_467, (1024, 512), (512, 1))
    assert_size_stride(permute_471, (512, 384), (384, 1))
    assert_size_stride(permute_474, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_475, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_105, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_476, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_477, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_482, (384, 512), (512, 1))
    assert_size_stride(permute_487, (384, 512), (512, 1))
    assert_size_stride(permute_492, (384, 512), (512, 1))
    assert_size_stride(permute_496, (512, 384), (384, 1))
    assert_size_stride(permute_499, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_500, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_107, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_501, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_502, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_507, (384, 512), (512, 1))
    assert_size_stride(permute_512, (384, 512), (512, 1))
    assert_size_stride(permute_517, (384, 512), (512, 1))
    assert_size_stride(permute_521, (512, 1024), (1024, 1))
    assert_size_stride(permute_525, (1024, 512), (512, 1))
    assert_size_stride(permute_529, (1024, 512), (512, 1))
    assert_size_stride(permute_533, (512, 384), (384, 1))
    assert_size_stride(permute_536, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_537, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_111, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_538, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_539, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_544, (384, 512), (512, 1))
    assert_size_stride(permute_549, (384, 512), (512, 1))
    assert_size_stride(permute_554, (384, 512), (512, 1))
    assert_size_stride(permute_558, (512, 384), (384, 1))
    assert_size_stride(permute_561, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_562, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_113, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_563, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_564, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_569, (384, 512), (512, 1))
    assert_size_stride(permute_574, (384, 512), (512, 1))
    assert_size_stride(permute_579, (384, 512), (512, 1))
    assert_size_stride(permute_583, (512, 1024), (1024, 1))
    assert_size_stride(permute_587, (1024, 512), (512, 1))
    assert_size_stride(permute_591, (1024, 512), (512, 1))
    assert_size_stride(permute_595, (512, 384), (384, 1))
    assert_size_stride(permute_598, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_599, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_117, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_600, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_601, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_606, (384, 512), (512, 1))
    assert_size_stride(permute_611, (384, 512), (512, 1))
    assert_size_stride(permute_616, (384, 512), (512, 1))
    assert_size_stride(permute_620, (512, 384), (384, 1))
    assert_size_stride(permute_623, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_624, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_119, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_625, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_626, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_631, (384, 512), (512, 1))
    assert_size_stride(permute_636, (384, 512), (512, 1))
    assert_size_stride(permute_641, (384, 512), (512, 1))
    assert_size_stride(permute_645, (512, 1024), (1024, 1))
    assert_size_stride(permute_649, (1024, 512), (512, 1))
    assert_size_stride(permute_653, (1024, 512), (512, 1))
    assert_size_stride(permute_657, (512, 384), (384, 1))
    assert_size_stride(permute_660, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_661, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_123, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_662, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_663, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_668, (384, 512), (512, 1))
    assert_size_stride(permute_673, (384, 512), (512, 1))
    assert_size_stride(permute_678, (384, 512), (512, 1))
    assert_size_stride(permute_682, (512, 384), (384, 1))
    assert_size_stride(permute_685, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_686, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_125, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_687, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_688, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_693, (384, 512), (512, 1))
    assert_size_stride(permute_698, (384, 512), (512, 1))
    assert_size_stride(permute_703, (384, 512), (512, 1))
    assert_size_stride(permute_707, (512, 1024), (1024, 1))
    assert_size_stride(permute_711, (1024, 512), (512, 1))
    assert_size_stride(permute_715, (1024, 512), (512, 1))
    assert_size_stride(permute_719, (512, 384), (384, 1))
    assert_size_stride(permute_722, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_723, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_129, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_724, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_725, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_730, (384, 512), (512, 1))
    assert_size_stride(permute_735, (384, 512), (512, 1))
    assert_size_stride(permute_740, (384, 512), (512, 1))
    assert_size_stride(permute_744, (512, 384), (384, 1))
    assert_size_stride(permute_747, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_748, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_131, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_750, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_751, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_756, (384, 512), (512, 1))
    assert_size_stride(permute_761, (384, 512), (512, 1))
    assert_size_stride(permute_766, (384, 512), (512, 1))
    assert_size_stride(permute_770, (512, 1024), (1024, 1))
    assert_size_stride(permute_774, (1024, 512), (512, 1))
    assert_size_stride(permute_778, (1024, 512), (512, 1))
    assert_size_stride(permute_782, (512, 384), (384, 1))
    assert_size_stride(permute_785, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_786, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_136, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_787, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_788, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_793, (384, 512), (512, 1))
    assert_size_stride(permute_798, (384, 512), (512, 1))
    assert_size_stride(permute_803, (384, 512), (512, 1))
    assert_size_stride(permute_807, (512, 1024), (1024, 1))
    assert_size_stride(permute_811, (1024, 512), (512, 1))
    assert_size_stride(permute_815, (1024, 512), (512, 1))
    assert_size_stride(permute_819, (512, 384), (384, 1))
    assert_size_stride(permute_822, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_823, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_140, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_824, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_825, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_830, (384, 512), (512, 1))
    assert_size_stride(permute_835, (384, 512), (512, 1))
    assert_size_stride(permute_840, (384, 512), (512, 1))
    assert_size_stride(permute_844, (512, 1024), (1024, 1))
    assert_size_stride(permute_848, (1024, 512), (512, 1))
    assert_size_stride(permute_852, (1024, 512), (512, 1))
    assert_size_stride(permute_856, (512, 384), (384, 1))
    assert_size_stride(permute_859, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_860, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_144, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_861, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_862, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_867, (384, 512), (512, 1))
    assert_size_stride(permute_872, (384, 512), (512, 1))
    assert_size_stride(permute_877, (384, 512), (512, 1))
    assert_size_stride(permute_881, (512, 1024), (1024, 1))
    assert_size_stride(permute_885, (1024, 512), (512, 1))
    assert_size_stride(permute_889, (1024, 512), (512, 1))
    assert_size_stride(permute_893, (512, 384), (384, 1))
    assert_size_stride(permute_896, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_897, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_148, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_898, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_899, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_904, (384, 512), (512, 1))
    assert_size_stride(permute_909, (384, 512), (512, 1))
    assert_size_stride(permute_914, (384, 512), (512, 1))
    assert_size_stride(permute_918, (512, 1024), (1024, 1))
    assert_size_stride(permute_922, (1024, 512), (512, 1))
    assert_size_stride(permute_926, (1024, 512), (512, 1))
    assert_size_stride(permute_930, (512, 384), (384, 1))
    assert_size_stride(permute_933, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_934, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_152, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_935, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_936, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_941, (384, 512), (512, 1))
    assert_size_stride(permute_946, (384, 512), (512, 1))
    assert_size_stride(permute_951, (384, 512), (512, 1))
    assert_size_stride(permute_955, (512, 1024), (1024, 1))
    assert_size_stride(permute_959, (1024, 512), (512, 1))
    assert_size_stride(permute_963, (1024, 512), (512, 1))
    assert_size_stride(permute_967, (512, 384), (384, 1))
    assert_size_stride(permute_970, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_971, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_156, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_972, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_973, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_978, (384, 512), (512, 1))
    assert_size_stride(permute_983, (384, 512), (512, 1))
    assert_size_stride(permute_988, (384, 512), (512, 1))
    assert_size_stride(permute_992, (512, 1024), (1024, 1))
    assert_size_stride(permute_996, (1024, 512), (512, 1))
    assert_size_stride(permute_1000, (1024, 512), (512, 1))
    assert_size_stride(permute_1004, (512, 384), (384, 1))
    assert_size_stride(permute_1007, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1008, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_160, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_1009, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_1010, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_1015, (384, 512), (512, 1))
    assert_size_stride(permute_1020, (384, 512), (512, 1))
    assert_size_stride(permute_1025, (384, 512), (512, 1))
    assert_size_stride(permute_1029, (512, 1024), (1024, 1))
    assert_size_stride(permute_1033, (1024, 512), (512, 1))
    assert_size_stride(permute_1037, (1024, 512), (512, 1))
    assert_size_stride(permute_1041, (512, 384), (384, 1))
    assert_size_stride(permute_1044, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1045, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_164, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_1047, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_1048, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_1053, (384, 512), (512, 1))
    assert_size_stride(permute_1058, (384, 512), (512, 1))
    assert_size_stride(permute_1063, (384, 512), (512, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 250112), (32014336, 250112, 1))
    assert_size_stride(tangents_3, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_4, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_5, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_6, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_7, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_8, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_9, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_10, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_11, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_12, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_13, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_14, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_15, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_16, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_17, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_18, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_19, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_20, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_21, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_22, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_23, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_24, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_25, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_26, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_27, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_28, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_29, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_30, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_31, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_32, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_33, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_34, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_35, (1, 128, 512), (65536, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((128, 250112), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 32014336, grid=grid(32014336), stream=stream0)
        buf1 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_192, buf1, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf4 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_192, tangents_1, convert_element_type_7, buf4, 512, 62528, grid=grid(512), stream=stream0)
        buf5 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3.run(buf4, buf5, 128, 4, grid=grid(128), stream=stream0)
        buf3 = empty((128, 250112), device='cuda', dtype=torch.float32)
        buf6 = reinterpret_tensor(buf3, (1, 128, 250112), (32014336, 250112, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf6, tangents_2, buf0, primals_192, tangents_1, convert_element_type_7, sub_30, buf5, 32014336, grid=grid(32014336), stream=stream0)
        del buf0
        del buf5
        del convert_element_type_7
        del primals_192
        del sub_30
        del tangents_1
        del tangents_2
        buf7 = empty((250112, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (250112, 128), (1, 250112), 0), view_578, out=buf7)
        del view_578
        buf8 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 250112), (250112, 1), 0), permute_269, out=buf8)
        del buf6
        del permute_269
        buf9 = reinterpret_tensor(buf4, (1, 1, 512), (512, 512, 1), 0); del buf4  # reuse
        # Source Nodes: [hidden_states_230], Original ATen: [aten.mul, aten.native_dropout_backward, aten.sum]
        triton_per_fused_mul_native_dropout_backward_sum_5.run(buf8, getitem_167, add_144, rsqrt_41, buf9, 512, 128, grid=grid(512), stream=stream0)
        buf11 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf12 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_6.run(buf8, getitem_167, primals_42, add_144, rsqrt_41, getitem_165, buf11, buf12, 128, 512, grid=grid(128), stream=stream0)
        del add_144
        del getitem_165
        del getitem_167
        del primals_42
        del rsqrt_41
        buf13 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 128), (1, 512), 0), view_576, out=buf13)
        del view_576
        buf14 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (128, 512), (512, 1), 0), permute_273, out=buf14)
        del permute_273
        buf15 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf18 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Source Nodes: [add_118, hidden_gelu_15, mul_164], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf19, buf14, getitem_163, mm_141, tanh_15, mm_142, buf15, 131072, grid=grid(131072), stream=stream0)
        del getitem_163
        del mm_141
        del mm_142
        del tanh_15
        buf16 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (1024, 128), (1, 1024), 0), view_572, out=buf16)
        buf17 = reinterpret_tensor(buf12, (128, 512), (512, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (128, 1024), (1024, 1), 0), permute_277, out=buf17)
        del permute_277
        buf20 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (1024, 128), (1, 1024), 0), view_572, out=buf20)
        del view_572
        buf21 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (128, 1024), (1024, 1), 0), permute_281, out=buf21)
        del permute_281
        buf22 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_223], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf17, buf21, add_140, rsqrt_40, buf22, 512, 128, grid=grid(512), stream=stream0)
        buf24 = buf11; del buf11  # reuse
        buf25 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf24, buf17, buf21, primals_41, add_140, rsqrt_40, getitem_161, buf25, 128, 512, grid=grid(128), stream=stream0)
        del add_140
        del getitem_161
        del primals_41
        del rsqrt_40
        buf26 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (512, 128), (1, 512), 0), view_570, out=buf26)
        del view_570
        buf27 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 512), (512, 1), 0), permute_285, out=buf27)
        del permute_285
        buf28 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_288, reinterpret_tensor(buf27, (6, 128, 64), (64, 384, 1), 0), out=buf28)
        del permute_288
        buf29 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (6, 128, 64), (64, 384, 1), 0), permute_289, out=buf29)
        del permute_289
        buf33 = empty((98304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf33, 98304, grid=grid(98304), stream=stream0)
        buf36 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf29, getitem_159, alias_87, buf33, buf36, 768, 128, grid=grid(768), stream=stream0)
        del alias_87
        del getitem_159
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf33, buf36, 98304, grid=grid(98304), stream=stream0)
        buf38 = reinterpret_tensor(buf27, (6, 64, 128), (8192, 128, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_290, buf36, out=buf38)
        del permute_290
        buf39 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf36, permute_291, out=buf39)
        del permute_291
        buf40 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_34, buf28, buf40, 49152, grid=grid(49152), stream=stream0)
        del tangents_34
        buf41 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (384, 128), (1, 384), 0), view_233, out=buf41)
        buf42 = reinterpret_tensor(buf25, (128, 512), (512, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (128, 384), (384, 1), 0), permute_296, out=buf42)
        del permute_296
        buf43 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_33, buf38, buf43, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_33
        buf44 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (384, 128), (1, 384), 0), view_233, out=buf44)
        buf45 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (128, 384), (384, 1), 0), permute_301, out=buf45)
        del permute_301
        buf46 = reinterpret_tensor(buf43, (128, 384), (384, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf39, buf46, 49152, grid=grid(49152), stream=stream0)
        buf47 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (384, 128), (1, 384), 0), view_552, out=buf47)
        del view_552
        buf48 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, permute_306, out=buf48)
        del permute_306
        buf49 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_219], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf48, add_137, rsqrt_39, buf49, 512, 128, grid=grid(512), stream=stream0)
        buf51 = buf24; del buf24  # reuse
        buf52 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf51, buf48, primals_40, add_137, rsqrt_39, getitem_157, buf52, 128, 512, grid=grid(128), stream=stream0)
        del add_137
        del getitem_157
        del primals_40
        del rsqrt_39
        buf53 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (512, 128), (1, 512), 0), view_550, out=buf53)
        del view_550
        buf54 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (128, 512), (512, 1), 0), permute_310, out=buf54)
        del permute_310
        buf55 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_313, reinterpret_tensor(buf54, (6, 128, 64), (64, 384, 1), 0), out=buf55)
        del permute_313
        buf56 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (6, 128, 64), (64, 384, 1), 0), permute_314, out=buf56)
        del permute_314
        buf59 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf59, 98304, grid=grid(98304), stream=stream0)
        buf62 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf56, getitem_155, alias_89, buf59, buf62, 768, 128, grid=grid(768), stream=stream0)
        del alias_89
        del getitem_155
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf59, buf62, 98304, grid=grid(98304), stream=stream0)
        buf64 = reinterpret_tensor(buf54, (6, 64, 128), (8192, 128, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_315, buf62, out=buf64)
        del permute_315
        buf65 = reinterpret_tensor(buf38, (6, 128, 64), (8192, 64, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf62, permute_316, out=buf65)
        del permute_316
        buf66 = reinterpret_tensor(buf28, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_32, buf55, buf66, 49152, grid=grid(49152), stream=stream0)
        del tangents_32
        buf67 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (384, 128), (1, 384), 0), view_532, out=buf67)
        buf68 = reinterpret_tensor(buf52, (128, 512), (512, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (128, 384), (384, 1), 0), permute_321, out=buf68)
        del permute_321
        buf69 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_31, buf64, buf69, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_31
        buf70 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (384, 128), (1, 384), 0), view_532, out=buf70)
        buf71 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (128, 384), (384, 1), 0), permute_326, out=buf71)
        del permute_326
        buf72 = reinterpret_tensor(buf69, (128, 384), (384, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf65, buf72, 49152, grid=grid(49152), stream=stream0)
        buf73 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (384, 128), (1, 384), 0), view_532, out=buf73)
        del view_532
        buf74 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf72, permute_331, out=buf74)
        del permute_331
        buf75 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_214], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf68, buf71, buf74, add_134, rsqrt_38, buf75, 512, 128, grid=grid(512), stream=stream0)
        buf77 = buf51; del buf51  # reuse
        buf78 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf77, buf68, buf71, buf74, primals_39, add_134, rsqrt_38, getitem_153, buf78, 128, 512, grid=grid(128), stream=stream0)
        del add_134
        del getitem_153
        del primals_39
        del rsqrt_38
        buf79 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 128), (1, 512), 0), view_530, out=buf79)
        del view_530
        buf80 = reinterpret_tensor(buf19, (128, 1024), (1024, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (128, 512), (512, 1), 0), permute_335, out=buf80)
        del permute_335
        buf81 = buf15; del buf15  # reuse
        buf84 = reinterpret_tensor(buf14, (1, 128, 1024), (131072, 1024, 1), 0); del buf14  # reuse
        buf85 = buf84; del buf84  # reuse
        # Source Nodes: [add_110, hidden_gelu_14, mul_153], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf85, buf80, getitem_151, mm_130, tanh_14, mm_131, buf81, 131072, grid=grid(131072), stream=stream0)
        del getitem_151
        del mm_130
        del mm_131
        del tanh_14
        buf82 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (1024, 128), (1, 1024), 0), view_526, out=buf82)
        buf83 = reinterpret_tensor(buf78, (128, 512), (512, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (128, 1024), (1024, 1), 0), permute_339, out=buf83)
        del permute_339
        buf86 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1024, 128), (1, 1024), 0), view_526, out=buf86)
        del view_526
        buf87 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (128, 1024), (1024, 1), 0), permute_343, out=buf87)
        del permute_343
        buf88 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_207], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf83, buf87, add_130, rsqrt_37, buf88, 512, 128, grid=grid(512), stream=stream0)
        buf90 = buf77; del buf77  # reuse
        buf91 = reinterpret_tensor(buf71, (1, 128, 512), (65536, 512, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf90, buf83, buf87, primals_38, add_130, rsqrt_37, getitem_149, buf91, 128, 512, grid=grid(128), stream=stream0)
        del add_130
        del getitem_149
        del primals_38
        del rsqrt_37
        buf92 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (512, 128), (1, 512), 0), view_524, out=buf92)
        del view_524
        buf93 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 512), (512, 1), 0), permute_347, out=buf93)
        del permute_347
        buf94 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_350, reinterpret_tensor(buf93, (6, 128, 64), (64, 384, 1), 0), out=buf94)
        del permute_350
        buf95 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (6, 128, 64), (64, 384, 1), 0), permute_351, out=buf95)
        del permute_351
        buf98 = reinterpret_tensor(buf56, (98304, ), (1, ), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf98, 98304, grid=grid(98304), stream=stream0)
        buf101 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf95, getitem_147, alias_93, buf98, buf101, 768, 128, grid=grid(768), stream=stream0)
        del alias_93
        del getitem_147
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf98, buf101, 98304, grid=grid(98304), stream=stream0)
        buf103 = reinterpret_tensor(buf93, (6, 64, 128), (8192, 128, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_352, buf101, out=buf103)
        del permute_352
        buf104 = reinterpret_tensor(buf64, (6, 128, 64), (8192, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf101, permute_353, out=buf104)
        del permute_353
        buf105 = reinterpret_tensor(buf55, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_30, buf94, buf105, 49152, grid=grid(49152), stream=stream0)
        del tangents_30
        buf106 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (384, 128), (1, 384), 0), view_233, out=buf106)
        buf107 = reinterpret_tensor(buf91, (128, 512), (512, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (128, 384), (384, 1), 0), permute_358, out=buf107)
        del permute_358
        buf108 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_29, buf103, buf108, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_29
        buf109 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (384, 128), (1, 384), 0), view_233, out=buf109)
        buf110 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (128, 384), (384, 1), 0), permute_363, out=buf110)
        del permute_363
        buf111 = reinterpret_tensor(buf108, (128, 384), (384, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf104, buf111, 49152, grid=grid(49152), stream=stream0)
        buf112 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (384, 128), (1, 384), 0), view_506, out=buf112)
        del view_506
        buf113 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf111, permute_368, out=buf113)
        del permute_368
        buf114 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_203], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf113, add_127, rsqrt_36, buf114, 512, 128, grid=grid(512), stream=stream0)
        buf116 = buf90; del buf90  # reuse
        buf117 = reinterpret_tensor(buf68, (1, 128, 512), (65536, 512, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf116, buf113, primals_37, add_127, rsqrt_36, getitem_145, buf117, 128, 512, grid=grid(128), stream=stream0)
        del add_127
        del getitem_145
        del primals_37
        del rsqrt_36
        buf118 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (512, 128), (1, 512), 0), view_504, out=buf118)
        del view_504
        buf119 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (128, 512), (512, 1), 0), permute_372, out=buf119)
        del permute_372
        buf120 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_375, reinterpret_tensor(buf119, (6, 128, 64), (64, 384, 1), 0), out=buf120)
        del permute_375
        buf121 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (6, 128, 64), (64, 384, 1), 0), permute_376, out=buf121)
        del permute_376
        buf124 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf124, 98304, grid=grid(98304), stream=stream0)
        buf127 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf121, getitem_143, alias_95, buf124, buf127, 768, 128, grid=grid(768), stream=stream0)
        del alias_95
        del getitem_143
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf124, buf127, 98304, grid=grid(98304), stream=stream0)
        buf129 = reinterpret_tensor(buf119, (6, 64, 128), (8192, 128, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_377, buf127, out=buf129)
        del permute_377
        buf130 = reinterpret_tensor(buf103, (6, 128, 64), (8192, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf127, permute_378, out=buf130)
        del permute_378
        buf131 = reinterpret_tensor(buf94, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_28, buf120, buf131, 49152, grid=grid(49152), stream=stream0)
        del tangents_28
        buf132 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (384, 128), (1, 384), 0), view_486, out=buf132)
        buf133 = reinterpret_tensor(buf117, (128, 512), (512, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (128, 384), (384, 1), 0), permute_383, out=buf133)
        del permute_383
        buf134 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_27, buf129, buf134, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_27
        buf135 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (384, 128), (1, 384), 0), view_486, out=buf135)
        buf136 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (128, 384), (384, 1), 0), permute_388, out=buf136)
        del permute_388
        buf137 = reinterpret_tensor(buf134, (128, 384), (384, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf130, buf137, 49152, grid=grid(49152), stream=stream0)
        buf138 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (384, 128), (1, 384), 0), view_486, out=buf138)
        del view_486
        buf139 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf137, permute_393, out=buf139)
        del permute_393
        buf140 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_198], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf133, buf136, buf139, add_124, rsqrt_35, buf140, 512, 128, grid=grid(512), stream=stream0)
        buf142 = buf116; del buf116  # reuse
        buf143 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf142, buf133, buf136, buf139, primals_36, add_124, rsqrt_35, getitem_141, buf143, 128, 512, grid=grid(128), stream=stream0)
        del add_124
        del getitem_141
        del primals_36
        del rsqrt_35
        buf144 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 128), (1, 512), 0), view_484, out=buf144)
        del view_484
        buf145 = reinterpret_tensor(buf85, (128, 1024), (1024, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (128, 512), (512, 1), 0), permute_397, out=buf145)
        del permute_397
        buf146 = buf81; del buf81  # reuse
        buf149 = reinterpret_tensor(buf80, (1, 128, 1024), (131072, 1024, 1), 0); del buf80  # reuse
        buf150 = buf149; del buf149  # reuse
        # Source Nodes: [add_102, hidden_gelu_13, mul_142], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf150, buf145, getitem_139, mm_119, tanh_13, mm_120, buf146, 131072, grid=grid(131072), stream=stream0)
        del getitem_139
        del mm_119
        del mm_120
        del tanh_13
        buf147 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (1024, 128), (1, 1024), 0), view_480, out=buf147)
        buf148 = reinterpret_tensor(buf143, (128, 512), (512, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (128, 1024), (1024, 1), 0), permute_401, out=buf148)
        del permute_401
        buf151 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (1024, 128), (1, 1024), 0), view_480, out=buf151)
        del view_480
        buf152 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (128, 1024), (1024, 1), 0), permute_405, out=buf152)
        del permute_405
        buf153 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_191], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf148, buf152, add_120, rsqrt_34, buf153, 512, 128, grid=grid(512), stream=stream0)
        buf155 = buf142; del buf142  # reuse
        buf156 = reinterpret_tensor(buf136, (1, 128, 512), (65536, 512, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf155, buf148, buf152, primals_35, add_120, rsqrt_34, getitem_137, buf156, 128, 512, grid=grid(128), stream=stream0)
        del add_120
        del getitem_137
        del primals_35
        del rsqrt_34
        buf157 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (512, 128), (1, 512), 0), view_478, out=buf157)
        del view_478
        buf158 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 512), (512, 1), 0), permute_409, out=buf158)
        del permute_409
        buf159 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_412, reinterpret_tensor(buf158, (6, 128, 64), (64, 384, 1), 0), out=buf159)
        del permute_412
        buf160 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (6, 128, 64), (64, 384, 1), 0), permute_413, out=buf160)
        del permute_413
        buf163 = reinterpret_tensor(buf121, (98304, ), (1, ), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf163, 98304, grid=grid(98304), stream=stream0)
        buf166 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf160, getitem_135, alias_99, buf163, buf166, 768, 128, grid=grid(768), stream=stream0)
        del alias_99
        del getitem_135
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf163, buf166, 98304, grid=grid(98304), stream=stream0)
        buf168 = reinterpret_tensor(buf158, (6, 64, 128), (8192, 128, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_414, buf166, out=buf168)
        del permute_414
        buf169 = reinterpret_tensor(buf129, (6, 128, 64), (8192, 64, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf166, permute_415, out=buf169)
        del permute_415
        buf170 = reinterpret_tensor(buf120, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_26, buf159, buf170, 49152, grid=grid(49152), stream=stream0)
        del tangents_26
        buf171 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (384, 128), (1, 384), 0), view_233, out=buf171)
        buf172 = reinterpret_tensor(buf156, (128, 512), (512, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (128, 384), (384, 1), 0), permute_420, out=buf172)
        del permute_420
        buf173 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_25, buf168, buf173, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_25
        buf174 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (384, 128), (1, 384), 0), view_233, out=buf174)
        buf175 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (128, 384), (384, 1), 0), permute_425, out=buf175)
        del permute_425
        buf176 = reinterpret_tensor(buf173, (128, 384), (384, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf169, buf176, 49152, grid=grid(49152), stream=stream0)
        buf177 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (384, 128), (1, 384), 0), view_460, out=buf177)
        del view_460
        buf178 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf176, permute_430, out=buf178)
        del permute_430
        buf179 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_187], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf178, add_117, rsqrt_33, buf179, 512, 128, grid=grid(512), stream=stream0)
        buf181 = buf155; del buf155  # reuse
        buf182 = reinterpret_tensor(buf133, (1, 128, 512), (65536, 512, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf181, buf178, primals_34, add_117, rsqrt_33, getitem_133, buf182, 128, 512, grid=grid(128), stream=stream0)
        del add_117
        del getitem_133
        del primals_34
        del rsqrt_33
        buf183 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (512, 128), (1, 512), 0), view_458, out=buf183)
        del view_458
        buf184 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (128, 512), (512, 1), 0), permute_434, out=buf184)
        del permute_434
        buf185 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_437, reinterpret_tensor(buf184, (6, 128, 64), (64, 384, 1), 0), out=buf185)
        del permute_437
        buf186 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (6, 128, 64), (64, 384, 1), 0), permute_438, out=buf186)
        del permute_438
        buf189 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf189, 98304, grid=grid(98304), stream=stream0)
        buf192 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf186, getitem_131, alias_101, buf189, buf192, 768, 128, grid=grid(768), stream=stream0)
        del alias_101
        del getitem_131
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf189, buf192, 98304, grid=grid(98304), stream=stream0)
        buf194 = reinterpret_tensor(buf184, (6, 64, 128), (8192, 128, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_439, buf192, out=buf194)
        del permute_439
        buf195 = reinterpret_tensor(buf168, (6, 128, 64), (8192, 64, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf192, permute_440, out=buf195)
        del permute_440
        buf196 = reinterpret_tensor(buf159, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_24, buf185, buf196, 49152, grid=grid(49152), stream=stream0)
        del tangents_24
        buf197 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (384, 128), (1, 384), 0), view_440, out=buf197)
        buf198 = reinterpret_tensor(buf182, (128, 512), (512, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 384), (384, 1), 0), permute_445, out=buf198)
        del permute_445
        buf199 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_23, buf194, buf199, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_23
        buf200 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (384, 128), (1, 384), 0), view_440, out=buf200)
        buf201 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (128, 384), (384, 1), 0), permute_450, out=buf201)
        del permute_450
        buf202 = reinterpret_tensor(buf199, (128, 384), (384, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf195, buf202, 49152, grid=grid(49152), stream=stream0)
        buf203 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (384, 128), (1, 384), 0), view_440, out=buf203)
        del view_440
        buf204 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, permute_455, out=buf204)
        del permute_455
        buf205 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_182], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf198, buf201, buf204, add_114, rsqrt_32, buf205, 512, 128, grid=grid(512), stream=stream0)
        buf207 = buf181; del buf181  # reuse
        buf208 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf207, buf198, buf201, buf204, primals_33, add_114, rsqrt_32, getitem_129, buf208, 128, 512, grid=grid(128), stream=stream0)
        del add_114
        del getitem_129
        del primals_33
        del rsqrt_32
        buf209 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (512, 128), (1, 512), 0), view_438, out=buf209)
        del view_438
        buf210 = reinterpret_tensor(buf150, (128, 1024), (1024, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 512), (512, 1), 0), permute_459, out=buf210)
        del permute_459
        buf211 = buf146; del buf146  # reuse
        buf214 = reinterpret_tensor(buf145, (1, 128, 1024), (131072, 1024, 1), 0); del buf145  # reuse
        buf215 = buf214; del buf214  # reuse
        # Source Nodes: [add_94, hidden_gelu_12, mul_131], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf215, buf210, getitem_127, mm_108, tanh_12, mm_109, buf211, 131072, grid=grid(131072), stream=stream0)
        del getitem_127
        del mm_108
        del mm_109
        del tanh_12
        buf212 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (1024, 128), (1, 1024), 0), view_434, out=buf212)
        buf213 = reinterpret_tensor(buf208, (128, 512), (512, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (128, 1024), (1024, 1), 0), permute_463, out=buf213)
        del permute_463
        buf216 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (1024, 128), (1, 1024), 0), view_434, out=buf216)
        del view_434
        buf217 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 1024), (1024, 1), 0), permute_467, out=buf217)
        del permute_467
        buf218 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_175], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf213, buf217, add_110, rsqrt_31, buf218, 512, 128, grid=grid(512), stream=stream0)
        buf220 = buf207; del buf207  # reuse
        buf221 = reinterpret_tensor(buf201, (1, 128, 512), (65536, 512, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf220, buf213, buf217, primals_32, add_110, rsqrt_31, getitem_125, buf221, 128, 512, grid=grid(128), stream=stream0)
        del add_110
        del getitem_125
        del primals_32
        del rsqrt_31
        buf222 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (512, 128), (1, 512), 0), view_432, out=buf222)
        del view_432
        buf223 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (128, 512), (512, 1), 0), permute_471, out=buf223)
        del permute_471
        buf224 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_474, reinterpret_tensor(buf223, (6, 128, 64), (64, 384, 1), 0), out=buf224)
        del permute_474
        buf225 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (6, 128, 64), (64, 384, 1), 0), permute_475, out=buf225)
        del permute_475
        buf228 = reinterpret_tensor(buf186, (98304, ), (1, ), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf228, 98304, grid=grid(98304), stream=stream0)
        buf231 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf225, getitem_123, alias_105, buf228, buf231, 768, 128, grid=grid(768), stream=stream0)
        del alias_105
        del getitem_123
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf228, buf231, 98304, grid=grid(98304), stream=stream0)
        buf233 = reinterpret_tensor(buf223, (6, 64, 128), (8192, 128, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_476, buf231, out=buf233)
        del permute_476
        buf234 = reinterpret_tensor(buf194, (6, 128, 64), (8192, 64, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf231, permute_477, out=buf234)
        del permute_477
        buf235 = reinterpret_tensor(buf185, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_22, buf224, buf235, 49152, grid=grid(49152), stream=stream0)
        del tangents_22
        buf236 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (384, 128), (1, 384), 0), view_233, out=buf236)
        buf237 = reinterpret_tensor(buf221, (128, 512), (512, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (128, 384), (384, 1), 0), permute_482, out=buf237)
        del permute_482
        buf238 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_21, buf233, buf238, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_21
        buf239 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (384, 128), (1, 384), 0), view_233, out=buf239)
        buf240 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (128, 384), (384, 1), 0), permute_487, out=buf240)
        del permute_487
        buf242 = reinterpret_tensor(buf238, (128, 384), (384, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf234, buf242, 49152, grid=grid(49152), stream=stream0)
        buf244 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf242, permute_492, out=buf244)
        del permute_492
        buf247 = buf220; del buf220  # reuse
        buf248 = reinterpret_tensor(buf198, (1, 128, 512), (65536, 512, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf247, buf244, primals_31, add_107, rsqrt_30, getitem_121, buf248, 128, 512, grid=grid(128), stream=stream0)
        del getitem_121
        del primals_31
        buf250 = reinterpret_tensor(buf234, (128, 384), (384, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (128, 512), (512, 1), 0), permute_496, out=buf250)
        del permute_496
        buf251 = reinterpret_tensor(buf233, (6, 128, 64), (8192, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_499, reinterpret_tensor(buf250, (6, 128, 64), (64, 384, 1), 0), out=buf251)
        del permute_499
        buf262 = reinterpret_tensor(buf224, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_20, buf251, buf262, 49152, grid=grid(49152), stream=stream0)
        del tangents_20
        buf264 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (128, 384), (384, 1), 0), permute_507, out=buf264)
        del permute_507
        buf252 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf250, (6, 128, 64), (64, 384, 1), 0), permute_500, out=buf252)
        del permute_500
        buf255 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf255, 98304, grid=grid(98304), stream=stream0)
        buf258 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf252, getitem_119, alias_107, buf255, buf258, 768, 128, grid=grid(768), stream=stream0)
        del alias_107
        del getitem_119
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf255, buf258, 98304, grid=grid(98304), stream=stream0)
        buf260 = reinterpret_tensor(buf250, (6, 64, 128), (8192, 128, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_501, buf258, out=buf260)
        del permute_501
        buf265 = reinterpret_tensor(buf251, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_19, buf260, buf265, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_19
        buf267 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (128, 384), (384, 1), 0), permute_512, out=buf267)
        del permute_512
        buf261 = reinterpret_tensor(buf260, (6, 128, 64), (8192, 64, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf258, permute_502, out=buf261)
        del permute_502
        buf268 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf261, buf268, 49152, grid=grid(49152), stream=stream0)
        buf270 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf268, permute_517, out=buf270)
        del permute_517
        buf273 = buf247; del buf247  # reuse
        buf274 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf273, buf264, buf267, buf270, primals_30, add_104, rsqrt_29, getitem_117, buf274, 128, 512, grid=grid(128), stream=stream0)
        del getitem_117
        del primals_30
        buf276 = reinterpret_tensor(buf215, (128, 1024), (1024, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (128, 512), (512, 1), 0), permute_521, out=buf276)
        del permute_521
        buf277 = buf211; del buf211  # reuse
        buf280 = reinterpret_tensor(buf210, (1, 128, 1024), (131072, 1024, 1), 0); del buf210  # reuse
        buf281 = buf280; del buf280  # reuse
        # Source Nodes: [add_86, hidden_gelu_11, mul_120], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf281, buf276, getitem_115, mm_97, tanh_11, mm_98, buf277, 131072, grid=grid(131072), stream=stream0)
        del getitem_115
        del mm_97
        del mm_98
        del tanh_11
        buf279 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (128, 1024), (1024, 1), 0), permute_525, out=buf279)
        del permute_525
        buf283 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf281, (128, 1024), (1024, 1), 0), permute_529, out=buf283)
        del permute_529
        buf286 = buf273; del buf273  # reuse
        buf287 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf286, buf279, buf283, primals_29, add_100, rsqrt_28, getitem_113, buf287, 128, 512, grid=grid(128), stream=stream0)
        del getitem_113
        del primals_29
        buf289 = reinterpret_tensor(buf261, (128, 384), (384, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (128, 512), (512, 1), 0), permute_533, out=buf289)
        del permute_533
        buf290 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_536, reinterpret_tensor(buf289, (6, 128, 64), (64, 384, 1), 0), out=buf290)
        del permute_536
        buf301 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_18, buf290, buf301, 49152, grid=grid(49152), stream=stream0)
        del tangents_18
        buf303 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (128, 384), (384, 1), 0), permute_544, out=buf303)
        del permute_544
        buf291 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (6, 128, 64), (64, 384, 1), 0), permute_537, out=buf291)
        del permute_537
        buf294 = reinterpret_tensor(buf252, (98304, ), (1, ), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf294, 98304, grid=grid(98304), stream=stream0)
        buf297 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf291, getitem_111, alias_111, buf294, buf297, 768, 128, grid=grid(768), stream=stream0)
        del alias_111
        del getitem_111
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf294, buf297, 98304, grid=grid(98304), stream=stream0)
        buf299 = reinterpret_tensor(buf289, (6, 64, 128), (8192, 128, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_538, buf297, out=buf299)
        del permute_538
        buf304 = reinterpret_tensor(buf290, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_17, buf299, buf304, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_17
        buf306 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (128, 384), (384, 1), 0), permute_549, out=buf306)
        del permute_549
        buf300 = reinterpret_tensor(buf299, (6, 128, 64), (8192, 64, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf297, permute_539, out=buf300)
        del permute_539
        buf307 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf300, buf307, 49152, grid=grid(49152), stream=stream0)
        buf309 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf307, permute_554, out=buf309)
        del permute_554
        buf312 = buf286; del buf286  # reuse
        buf313 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf312, buf309, primals_28, add_97, rsqrt_27, getitem_109, buf313, 128, 512, grid=grid(128), stream=stream0)
        del getitem_109
        del primals_28
        buf315 = reinterpret_tensor(buf300, (128, 384), (384, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (128, 512), (512, 1), 0), permute_558, out=buf315)
        del permute_558
        buf316 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_561, reinterpret_tensor(buf315, (6, 128, 64), (64, 384, 1), 0), out=buf316)
        del permute_561
        buf327 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_16, buf316, buf327, 49152, grid=grid(49152), stream=stream0)
        del tangents_16
        buf329 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (128, 384), (384, 1), 0), permute_569, out=buf329)
        del permute_569
        buf317 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (6, 128, 64), (64, 384, 1), 0), permute_562, out=buf317)
        del permute_562
        buf320 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf320, 98304, grid=grid(98304), stream=stream0)
        buf323 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf317, getitem_107, alias_113, buf320, buf323, 768, 128, grid=grid(768), stream=stream0)
        del alias_113
        del getitem_107
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf320, buf323, 98304, grid=grid(98304), stream=stream0)
        buf325 = reinterpret_tensor(buf315, (6, 64, 128), (8192, 128, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_563, buf323, out=buf325)
        del permute_563
        buf330 = reinterpret_tensor(buf316, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_15, buf325, buf330, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_15
        buf332 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (128, 384), (384, 1), 0), permute_574, out=buf332)
        del permute_574
        buf326 = reinterpret_tensor(buf325, (6, 128, 64), (8192, 64, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf323, permute_564, out=buf326)
        del permute_564
        buf333 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf326, buf333, 49152, grid=grid(49152), stream=stream0)
        buf335 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf333, permute_579, out=buf335)
        del permute_579
        buf338 = buf312; del buf312  # reuse
        buf339 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf338, buf329, buf332, buf335, primals_27, add_94, rsqrt_26, getitem_105, buf339, 128, 512, grid=grid(128), stream=stream0)
        del getitem_105
        del primals_27
        buf341 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (128, 512), (512, 1), 0), permute_583, out=buf341)
        del permute_583
        buf342 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf345 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf346 = buf345; del buf345  # reuse
        # Source Nodes: [add_78, hidden_gelu_10, mul_109], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf346, buf341, getitem_103, mm_86, tanh_10, mm_87, buf342, 131072, grid=grid(131072), stream=stream0)
        del getitem_103
        del mm_86
        del mm_87
        del tanh_10
        buf344 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (128, 1024), (1024, 1), 0), permute_587, out=buf344)
        del permute_587
        buf348 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (128, 1024), (1024, 1), 0), permute_591, out=buf348)
        del permute_591
        buf351 = buf338; del buf338  # reuse
        buf352 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf351, buf344, buf348, primals_26, add_90, rsqrt_25, getitem_101, buf352, 128, 512, grid=grid(128), stream=stream0)
        del getitem_101
        del primals_26
        buf354 = reinterpret_tensor(buf326, (128, 384), (384, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (128, 512), (512, 1), 0), permute_595, out=buf354)
        del permute_595
        buf355 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_598, reinterpret_tensor(buf354, (6, 128, 64), (64, 384, 1), 0), out=buf355)
        del permute_598
        buf366 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_14, buf355, buf366, 49152, grid=grid(49152), stream=stream0)
        del tangents_14
        buf368 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (128, 384), (384, 1), 0), permute_606, out=buf368)
        del permute_606
        buf356 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf354, (6, 128, 64), (64, 384, 1), 0), permute_599, out=buf356)
        del permute_599
        buf359 = reinterpret_tensor(buf317, (98304, ), (1, ), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf359, 98304, grid=grid(98304), stream=stream0)
        buf362 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf356, getitem_99, alias_117, buf359, buf362, 768, 128, grid=grid(768), stream=stream0)
        del alias_117
        del getitem_99
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf359, buf362, 98304, grid=grid(98304), stream=stream0)
        buf364 = reinterpret_tensor(buf354, (6, 64, 128), (8192, 128, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_600, buf362, out=buf364)
        del permute_600
        buf369 = reinterpret_tensor(buf355, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_13, buf364, buf369, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_13
        buf371 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (128, 384), (384, 1), 0), permute_611, out=buf371)
        del permute_611
        buf365 = reinterpret_tensor(buf364, (6, 128, 64), (8192, 64, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf362, permute_601, out=buf365)
        del permute_601
        buf372 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf365, buf372, 49152, grid=grid(49152), stream=stream0)
        buf374 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf372, permute_616, out=buf374)
        del permute_616
        buf377 = buf351; del buf351  # reuse
        buf378 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf377, buf374, primals_25, add_87, rsqrt_24, getitem_97, buf378, 128, 512, grid=grid(128), stream=stream0)
        del getitem_97
        del primals_25
        buf380 = reinterpret_tensor(buf365, (128, 384), (384, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (128, 512), (512, 1), 0), permute_620, out=buf380)
        del permute_620
        buf381 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_623, reinterpret_tensor(buf380, (6, 128, 64), (64, 384, 1), 0), out=buf381)
        del permute_623
        buf392 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_12, buf381, buf392, 49152, grid=grid(49152), stream=stream0)
        del tangents_12
        buf394 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (128, 384), (384, 1), 0), permute_631, out=buf394)
        del permute_631
        buf382 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf380, (6, 128, 64), (64, 384, 1), 0), permute_624, out=buf382)
        del permute_624
        buf385 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf385, 98304, grid=grid(98304), stream=stream0)
        buf388 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf382, getitem_95, alias_119, buf385, buf388, 768, 128, grid=grid(768), stream=stream0)
        del alias_119
        del getitem_95
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf385, buf388, 98304, grid=grid(98304), stream=stream0)
        buf390 = reinterpret_tensor(buf380, (6, 64, 128), (8192, 128, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_625, buf388, out=buf390)
        del permute_625
        buf395 = reinterpret_tensor(buf381, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_11, buf390, buf395, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_11
        buf397 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (128, 384), (384, 1), 0), permute_636, out=buf397)
        del permute_636
        buf391 = reinterpret_tensor(buf390, (6, 128, 64), (8192, 64, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf388, permute_626, out=buf391)
        del permute_626
        buf398 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf391, buf398, 49152, grid=grid(49152), stream=stream0)
        buf400 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf398, permute_641, out=buf400)
        del permute_641
        buf403 = buf377; del buf377  # reuse
        buf404 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf403, buf394, buf397, buf400, primals_24, add_84, rsqrt_23, getitem_93, buf404, 128, 512, grid=grid(128), stream=stream0)
        del getitem_93
        del primals_24
        buf406 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (128, 512), (512, 1), 0), permute_645, out=buf406)
        del permute_645
        buf407 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf410 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf411 = buf410; del buf410  # reuse
        # Source Nodes: [add_70, hidden_gelu_9, mul_98], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf411, buf406, getitem_91, mm_75, tanh_9, mm_76, buf407, 131072, grid=grid(131072), stream=stream0)
        del getitem_91
        del mm_75
        del mm_76
        del tanh_9
        buf409 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (128, 1024), (1024, 1), 0), permute_649, out=buf409)
        del permute_649
        buf413 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (128, 1024), (1024, 1), 0), permute_653, out=buf413)
        del permute_653
        buf416 = buf403; del buf403  # reuse
        buf417 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf416, buf409, buf413, primals_23, add_80, rsqrt_22, getitem_89, buf417, 128, 512, grid=grid(128), stream=stream0)
        del getitem_89
        del primals_23
        buf419 = reinterpret_tensor(buf391, (128, 384), (384, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (128, 512), (512, 1), 0), permute_657, out=buf419)
        del permute_657
        buf420 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_660, reinterpret_tensor(buf419, (6, 128, 64), (64, 384, 1), 0), out=buf420)
        del permute_660
        buf431 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_10, buf420, buf431, 49152, grid=grid(49152), stream=stream0)
        del tangents_10
        buf433 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf431, (128, 384), (384, 1), 0), permute_668, out=buf433)
        del permute_668
        buf421 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (6, 128, 64), (64, 384, 1), 0), permute_661, out=buf421)
        del permute_661
        buf424 = reinterpret_tensor(buf382, (98304, ), (1, ), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf424, 98304, grid=grid(98304), stream=stream0)
        buf427 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf421, getitem_87, alias_123, buf424, buf427, 768, 128, grid=grid(768), stream=stream0)
        del alias_123
        del getitem_87
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf424, buf427, 98304, grid=grid(98304), stream=stream0)
        buf429 = reinterpret_tensor(buf419, (6, 64, 128), (8192, 128, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_662, buf427, out=buf429)
        del permute_662
        buf434 = reinterpret_tensor(buf420, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_9, buf429, buf434, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_9
        buf436 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (128, 384), (384, 1), 0), permute_673, out=buf436)
        del permute_673
        buf430 = reinterpret_tensor(buf429, (6, 128, 64), (8192, 64, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf427, permute_663, out=buf430)
        del permute_663
        buf437 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf430, buf437, 49152, grid=grid(49152), stream=stream0)
        buf439 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf437, permute_678, out=buf439)
        del permute_678
        buf442 = buf416; del buf416  # reuse
        buf443 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf442, buf439, primals_22, add_77, rsqrt_21, getitem_85, buf443, 128, 512, grid=grid(128), stream=stream0)
        del getitem_85
        del primals_22
        buf445 = reinterpret_tensor(buf430, (128, 384), (384, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (128, 512), (512, 1), 0), permute_682, out=buf445)
        del permute_682
        buf446 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_685, reinterpret_tensor(buf445, (6, 128, 64), (64, 384, 1), 0), out=buf446)
        del permute_685
        buf457 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_8, buf446, buf457, 49152, grid=grid(49152), stream=stream0)
        del tangents_8
        buf459 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (128, 384), (384, 1), 0), permute_693, out=buf459)
        del permute_693
        buf447 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (6, 128, 64), (64, 384, 1), 0), permute_686, out=buf447)
        del permute_686
        buf450 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf450, 98304, grid=grid(98304), stream=stream0)
        buf453 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf447, getitem_83, alias_125, buf450, buf453, 768, 128, grid=grid(768), stream=stream0)
        del alias_125
        del getitem_83
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf450, buf453, 98304, grid=grid(98304), stream=stream0)
        buf455 = reinterpret_tensor(buf445, (6, 64, 128), (8192, 128, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_687, buf453, out=buf455)
        del permute_687
        buf460 = reinterpret_tensor(buf446, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_7, buf455, buf460, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_7
        buf462 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (128, 384), (384, 1), 0), permute_698, out=buf462)
        del permute_698
        buf456 = reinterpret_tensor(buf455, (6, 128, 64), (8192, 64, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf453, permute_688, out=buf456)
        del permute_688
        buf463 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf456, buf463, 49152, grid=grid(49152), stream=stream0)
        buf465 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf463, permute_703, out=buf465)
        del permute_703
        buf468 = buf442; del buf442  # reuse
        buf469 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf468, buf459, buf462, buf465, primals_21, add_74, rsqrt_20, getitem_81, buf469, 128, 512, grid=grid(128), stream=stream0)
        del getitem_81
        del primals_21
        buf471 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (128, 512), (512, 1), 0), permute_707, out=buf471)
        del permute_707
        buf472 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf475 = empty((1, 128, 1024), device='cuda', dtype=torch.float32)
        buf476 = buf475; del buf475  # reuse
        # Source Nodes: [add_62, hidden_gelu_8, mul_87], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf476, buf471, getitem_79, mm_64, tanh_8, mm_65, buf472, 131072, grid=grid(131072), stream=stream0)
        del buf471
        del getitem_79
        del mm_64
        del mm_65
        del tanh_8
        buf474 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf472, (128, 1024), (1024, 1), 0), permute_711, out=buf474)
        del permute_711
        buf478 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (128, 1024), (1024, 1), 0), permute_715, out=buf478)
        del permute_715
        buf481 = buf468; del buf468  # reuse
        buf482 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf481, buf474, buf478, primals_20, add_70, rsqrt_19, getitem_77, buf482, 128, 512, grid=grid(128), stream=stream0)
        del getitem_77
        del primals_20
        buf484 = reinterpret_tensor(buf456, (128, 384), (384, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (128, 512), (512, 1), 0), permute_719, out=buf484)
        del permute_719
        buf485 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_722, reinterpret_tensor(buf484, (6, 128, 64), (64, 384, 1), 0), out=buf485)
        del permute_722
        buf496 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_6, buf485, buf496, 49152, grid=grid(49152), stream=stream0)
        del tangents_6
        buf498 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (128, 384), (384, 1), 0), permute_730, out=buf498)
        del permute_730
        buf486 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf484, (6, 128, 64), (64, 384, 1), 0), permute_723, out=buf486)
        del permute_723
        buf489 = reinterpret_tensor(buf447, (98304, ), (1, ), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf489, 98304, grid=grid(98304), stream=stream0)
        buf492 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf486, getitem_75, alias_129, buf489, buf492, 768, 128, grid=grid(768), stream=stream0)
        del alias_129
        del getitem_75
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf489, buf492, 98304, grid=grid(98304), stream=stream0)
        buf494 = reinterpret_tensor(buf484, (6, 64, 128), (8192, 128, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_724, buf492, out=buf494)
        del permute_724
        buf499 = reinterpret_tensor(buf485, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_5, buf494, buf499, 768, 64, grid=grid(768, 64), stream=stream0)
        del buf494
        del tangents_5
        buf501 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (128, 384), (384, 1), 0), permute_735, out=buf501)
        del permute_735
        buf241 = reinterpret_tensor(buf107, (1, 128, 512), (65536, 512, 1), 0); del buf107  # reuse
        buf502 = buf241; del buf241  # reuse
        buf545 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf546 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_20.run(buf502, tangents_35, buf42, buf45, buf110, buf172, buf175, buf237, buf240, buf303, buf306, buf368, buf371, buf433, buf436, buf498, buf501, getitem_67, primals_17, add_59, rsqrt_16, getitem_65, buf545, buf546, 128, 512, grid=grid(128), stream=stream0)
        del buf110
        del buf172
        del buf175
        del buf237
        del buf240
        del buf303
        del buf306
        del buf368
        del buf371
        del buf42
        del buf433
        del buf436
        del buf45
        del buf498
        del buf501
        del getitem_65
        del primals_17
        del tangents_35
        buf243 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (384, 128), (1, 384), 0), view_414, out=buf243)
        del buf242
        del view_414
        buf245 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_171], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf244, add_107, rsqrt_30, buf245, 512, 128, grid=grid(512), stream=stream0)
        del add_107
        del buf244
        del rsqrt_30
        buf249 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (512, 128), (1, 512), 0), view_412, out=buf249)
        del buf248
        del view_412
        buf263 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (384, 128), (1, 384), 0), view_394, out=buf263)
        del buf262
        buf266 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (384, 128), (1, 384), 0), view_394, out=buf266)
        del buf265
        buf269 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (384, 128), (1, 384), 0), view_394, out=buf269)
        del buf268
        del view_394
        buf271 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_166], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf264, buf267, buf270, add_104, rsqrt_29, buf271, 512, 128, grid=grid(512), stream=stream0)
        del add_104
        del buf264
        del buf267
        del buf270
        del rsqrt_29
        buf275 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (512, 128), (1, 512), 0), view_392, out=buf275)
        del buf274
        del view_392
        buf278 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (1024, 128), (1, 1024), 0), view_388, out=buf278)
        del buf277
        buf282 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf281, (1024, 128), (1, 1024), 0), view_388, out=buf282)
        del buf281
        del view_388
        buf284 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_159], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf279, buf283, add_100, rsqrt_28, buf284, 512, 128, grid=grid(512), stream=stream0)
        del add_100
        del buf279
        del buf283
        del rsqrt_28
        buf288 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (512, 128), (1, 512), 0), view_386, out=buf288)
        del buf287
        del view_386
        buf302 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (384, 128), (1, 384), 0), view_233, out=buf302)
        del buf301
        buf305 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (384, 128), (1, 384), 0), view_233, out=buf305)
        del buf304
        buf308 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (384, 128), (1, 384), 0), view_368, out=buf308)
        del buf307
        del view_368
        buf310 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_155], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf309, add_97, rsqrt_27, buf310, 512, 128, grid=grid(512), stream=stream0)
        del add_97
        del buf309
        del rsqrt_27
        buf314 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (512, 128), (1, 512), 0), view_366, out=buf314)
        del buf313
        del view_366
        buf328 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (384, 128), (1, 384), 0), view_348, out=buf328)
        del buf327
        buf331 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (384, 128), (1, 384), 0), view_348, out=buf331)
        del buf330
        buf334 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (384, 128), (1, 384), 0), view_348, out=buf334)
        del buf333
        del view_348
        buf336 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_150], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf329, buf332, buf335, add_94, rsqrt_26, buf336, 512, 128, grid=grid(512), stream=stream0)
        del add_94
        del buf329
        del buf332
        del buf335
        del rsqrt_26
        buf340 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 128), (1, 512), 0), view_346, out=buf340)
        del buf339
        del view_346
        buf343 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (1024, 128), (1, 1024), 0), view_342, out=buf343)
        del buf342
        buf347 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (1024, 128), (1, 1024), 0), view_342, out=buf347)
        del buf346
        del view_342
        buf349 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_143], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf344, buf348, add_90, rsqrt_25, buf349, 512, 128, grid=grid(512), stream=stream0)
        del add_90
        del buf344
        del buf348
        del rsqrt_25
        buf353 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (512, 128), (1, 512), 0), view_340, out=buf353)
        del buf352
        del view_340
        buf367 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (384, 128), (1, 384), 0), view_233, out=buf367)
        del buf366
        buf370 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (384, 128), (1, 384), 0), view_233, out=buf370)
        del buf369
        buf373 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (384, 128), (1, 384), 0), view_322, out=buf373)
        del buf372
        del view_322
        buf375 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_139], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf374, add_87, rsqrt_24, buf375, 512, 128, grid=grid(512), stream=stream0)
        del add_87
        del buf374
        del rsqrt_24
        buf379 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (512, 128), (1, 512), 0), view_320, out=buf379)
        del buf378
        del view_320
        buf393 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (384, 128), (1, 384), 0), view_302, out=buf393)
        del buf392
        buf396 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (384, 128), (1, 384), 0), view_302, out=buf396)
        del buf395
        buf399 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf398, (384, 128), (1, 384), 0), view_302, out=buf399)
        del buf398
        del view_302
        buf401 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf394, buf397, buf400, add_84, rsqrt_23, buf401, 512, 128, grid=grid(512), stream=stream0)
        del add_84
        del buf394
        del buf397
        del buf400
        del rsqrt_23
        buf405 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (512, 128), (1, 512), 0), view_300, out=buf405)
        del buf404
        del view_300
        buf408 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (1024, 128), (1, 1024), 0), view_296, out=buf408)
        del buf407
        buf412 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (1024, 128), (1, 1024), 0), view_296, out=buf412)
        del view_296
        buf414 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_127], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf409, buf413, add_80, rsqrt_22, buf414, 512, 128, grid=grid(512), stream=stream0)
        del add_80
        del buf409
        del buf413
        del rsqrt_22
        buf418 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (512, 128), (1, 512), 0), view_294, out=buf418)
        del buf417
        del view_294
        buf432 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf431, (384, 128), (1, 384), 0), view_233, out=buf432)
        del buf431
        buf435 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (384, 128), (1, 384), 0), view_233, out=buf435)
        del buf434
        buf438 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (384, 128), (1, 384), 0), view_276, out=buf438)
        del buf437
        del view_276
        buf440 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_123], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf439, add_77, rsqrt_21, buf440, 512, 128, grid=grid(512), stream=stream0)
        del add_77
        del buf439
        del rsqrt_21
        buf444 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (512, 128), (1, 512), 0), view_274, out=buf444)
        del buf443
        del view_274
        buf458 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (384, 128), (1, 384), 0), view_256, out=buf458)
        del buf457
        buf461 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (384, 128), (1, 384), 0), view_256, out=buf461)
        buf464 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (384, 128), (1, 384), 0), view_256, out=buf464)
        del view_256
        buf466 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_118], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf459, buf462, buf465, add_74, rsqrt_20, buf466, 512, 128, grid=grid(512), stream=stream0)
        del add_74
        del buf459
        del buf462
        del buf465
        del rsqrt_20
        buf470 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (512, 128), (1, 512), 0), view_254, out=buf470)
        del buf469
        del view_254
        buf473 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf472, (1024, 128), (1, 1024), 0), view_250, out=buf473)
        buf477 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (1024, 128), (1, 1024), 0), view_250, out=buf477)
        del view_250
        buf479 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_111], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf474, buf478, add_70, rsqrt_19, buf479, 512, 128, grid=grid(512), stream=stream0)
        del add_70
        del rsqrt_19
        buf483 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (512, 128), (1, 512), 0), view_248, out=buf483)
        del view_248
        buf495 = reinterpret_tensor(buf463, (6, 128, 64), (8192, 64, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf492, permute_725, out=buf495)
        del permute_725
        buf497 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (384, 128), (1, 384), 0), view_233, out=buf497)
        buf500 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (384, 128), (1, 384), 0), view_233, out=buf500)
        del view_233
        buf503 = reinterpret_tensor(buf499, (128, 384), (384, 1), 0); del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf495, buf503, 49152, grid=grid(49152), stream=stream0)
        buf504 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf503, (384, 128), (1, 384), 0), view_230, out=buf504)
        del view_230
        buf505 = reinterpret_tensor(buf482, (128, 512), (512, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf503, permute_740, out=buf505)
        del permute_740
        buf506 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_107], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_16.run(buf505, add_66, rsqrt_18, buf506, 512, 128, grid=grid(512), stream=stream0)
        buf508 = buf481; del buf481  # reuse
        buf509 = reinterpret_tensor(buf478, (1, 128, 512), (65536, 512, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_17.run(buf508, buf505, primals_19, add_66, rsqrt_18, getitem_73, buf509, 128, 512, grid=grid(128), stream=stream0)
        del add_66
        del getitem_73
        del primals_19
        del rsqrt_18
        buf510 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (512, 128), (1, 512), 0), view_228, out=buf510)
        del view_228
        buf511 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (128, 512), (512, 1), 0), permute_744, out=buf511)
        del permute_744
        buf512 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_747, reinterpret_tensor(buf511, (6, 128, 64), (64, 384, 1), 0), out=buf512)
        del permute_747
        buf513 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf511, (6, 128, 64), (64, 384, 1), 0), permute_748, out=buf513)
        del permute_748
        buf516 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf516, 98304, grid=grid(98304), stream=stream0)
        buf519 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf513, getitem_71, alias_131, buf516, buf519, 768, 128, grid=grid(768), stream=stream0)
        del alias_131
        del getitem_71
        buf522 = reinterpret_tensor(buf513, (128, 128, 6), (128, 1, 16384), 0); del buf513  # reuse
        # Source Nodes: [loss], Original ATen: [aten.as_strided_scatter, aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_21.run(buf516, buf59, buf124, buf189, buf255, buf320, buf385, buf450, buf519, buf522, 98304, grid=grid(98304), stream=stream0)
        buf521 = empty((32, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_22.run(buf521, 192, grid=grid(192), stream=stream0)
        aten.index_put_(buf521, [add_63], buf522, True)
        del add_63
        buf525 = reinterpret_tensor(buf511, (6, 64, 128), (8192, 128, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_750, buf519, out=buf525)
        del permute_750
        buf526 = reinterpret_tensor(buf496, (6, 128, 64), (8192, 64, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf519, permute_751, out=buf526)
        del permute_751
        buf527 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_4, buf512, buf527, 49152, grid=grid(49152), stream=stream0)
        del tangents_4
        buf528 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (384, 128), (1, 384), 0), view_210, out=buf528)
        buf529 = reinterpret_tensor(buf509, (128, 512), (512, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (128, 384), (384, 1), 0), permute_756, out=buf529)
        del permute_756
        buf530 = buf527; del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_3, buf525, buf530, 768, 64, grid=grid(768, 64), stream=stream0)
        del tangents_3
        buf531 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (384, 128), (1, 384), 0), view_210, out=buf531)
        buf532 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (128, 384), (384, 1), 0), permute_761, out=buf532)
        del permute_761
        buf533 = reinterpret_tensor(buf530, (128, 384), (384, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf526, buf533, 49152, grid=grid(49152), stream=stream0)
        buf534 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (384, 128), (1, 384), 0), view_210, out=buf534)
        del view_210
        buf535 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf533, permute_766, out=buf535)
        del permute_766
        buf536 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf529, buf532, buf535, getitem_68, rsqrt_17, buf536, 512, 128, grid=grid(512), stream=stream0)
        buf538 = buf508; del buf508  # reuse
        buf540 = buf538; del buf538  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum]
        triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_23.run(buf540, buf529, buf532, buf535, primals_18, getitem_68, rsqrt_17, getitem_69, view_209, 128, 512, grid=grid(128), stream=stream0)
        del buf529
        del buf532
        del getitem_68
        del getitem_69
        del primals_18
        del rsqrt_17
        buf539 = empty((250112, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_24.run(buf539, 128057344, grid=grid(128057344), stream=stream0)
        aten.index_put_(buf539, [view_209], buf540, True)
        del view_209
        buf543 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_97], Original ATen: [aten.mul, aten.native_dropout_backward, aten.sum]
        triton_per_fused_mul_native_dropout_backward_sum_5.run(buf502, getitem_67, add_59, rsqrt_16, buf543, 512, 128, grid=grid(512), stream=stream0)
        del add_59
        del getitem_67
        del rsqrt_16
        buf547 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (512, 128), (1, 512), 0), view_207, out=buf547)
        del view_207
        buf548 = reinterpret_tensor(buf476, (128, 1024), (1024, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (128, 512), (512, 1), 0), permute_770, out=buf548)
        del permute_770
        buf549 = buf472; del buf472  # reuse
        buf552 = buf411; del buf411  # reuse
        buf553 = buf552; del buf552  # reuse
        # Source Nodes: [add_49, hidden_gelu_7, mul_70], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf553, buf548, getitem_63, mm_53, tanh_7, mm_54, buf549, 131072, grid=grid(131072), stream=stream0)
        del getitem_63
        del mm_53
        del mm_54
        del tanh_7
        buf550 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (1024, 128), (1, 1024), 0), view_203, out=buf550)
        buf551 = reinterpret_tensor(buf546, (128, 512), (512, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (128, 1024), (1024, 1), 0), permute_774, out=buf551)
        del permute_774
        buf554 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (1024, 128), (1, 1024), 0), view_203, out=buf554)
        del view_203
        buf555 = reinterpret_tensor(buf502, (128, 512), (512, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (128, 1024), (1024, 1), 0), permute_778, out=buf555)
        del permute_778
        buf556 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf551, buf555, add_55, rsqrt_15, buf556, 512, 128, grid=grid(512), stream=stream0)
        buf558 = buf545; del buf545  # reuse
        buf559 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf558, buf551, buf555, primals_16, add_55, rsqrt_15, getitem_61, buf559, 128, 512, grid=grid(128), stream=stream0)
        del add_55
        del getitem_61
        del primals_16
        del rsqrt_15
        buf560 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf559, (512, 128), (1, 512), 0), view_201, out=buf560)
        del view_201
        buf561 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf559, (128, 512), (512, 1), 0), permute_782, out=buf561)
        del permute_782
        buf562 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_785, reinterpret_tensor(buf561, (6, 128, 64), (64, 384, 1), 0), out=buf562)
        del permute_785
        buf563 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf561, (6, 128, 64), (64, 384, 1), 0), permute_786, out=buf563)
        del permute_786
        buf566 = reinterpret_tensor(buf522, (98304, ), (1, ), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf566, 98304, grid=grid(98304), stream=stream0)
        buf569 = reinterpret_tensor(buf59, (6, 128, 128), (16384, 128, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf563, getitem_59, alias_136, buf566, buf569, 768, 128, grid=grid(768), stream=stream0)
        del alias_136
        del getitem_59
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf566, buf569, 98304, grid=grid(98304), stream=stream0)
        buf571 = reinterpret_tensor(buf561, (6, 64, 128), (8192, 128, 1), 0); del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_787, buf569, out=buf571)
        del permute_787
        buf572 = reinterpret_tensor(buf525, (6, 128, 64), (8192, 64, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf569, permute_788, out=buf572)
        del permute_788
        buf573 = reinterpret_tensor(buf512, (128, 384), (384, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf562, buf573, 49152, grid=grid(49152), stream=stream0)
        buf574 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (384, 128), (1, 384), 0), view_183, out=buf574)
        buf575 = reinterpret_tensor(buf559, (128, 512), (512, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf573, permute_793, out=buf575)
        del permute_793
        buf576 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf571, (384, 128), (128, 1), 0), view_183, out=buf576)
        buf577 = buf555; del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf571, (128, 384), (1, 128), 0), permute_798, out=buf577)
        del permute_798
        buf578 = reinterpret_tensor(buf571, (128, 384), (384, 1), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf572, buf578, 49152, grid=grid(49152), stream=stream0)
        buf579 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf578, (384, 128), (1, 384), 0), view_183, out=buf579)
        del view_183
        buf580 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf578, permute_803, out=buf580)
        del permute_803
        buf581 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_85], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf575, buf577, buf580, add_52, rsqrt_14, buf581, 512, 128, grid=grid(512), stream=stream0)
        buf583 = buf558; del buf558  # reuse
        buf584 = reinterpret_tensor(buf535, (1, 128, 512), (65536, 512, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf583, buf575, buf577, buf580, primals_15, add_52, rsqrt_14, getitem_57, buf584, 128, 512, grid=grid(128), stream=stream0)
        del add_52
        del getitem_57
        del primals_15
        del rsqrt_14
        buf585 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf584, (512, 128), (1, 512), 0), view_181, out=buf585)
        del view_181
        buf586 = reinterpret_tensor(buf553, (128, 1024), (1024, 1), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf584, (128, 512), (512, 1), 0), permute_807, out=buf586)
        del permute_807
        buf587 = buf549; del buf549  # reuse
        buf590 = reinterpret_tensor(buf548, (1, 128, 1024), (131072, 1024, 1), 0); del buf548  # reuse
        buf591 = buf590; del buf590  # reuse
        # Source Nodes: [add_43, hidden_gelu_6, mul_61], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf591, buf586, getitem_55, mm_46, tanh_6, mm_47, buf587, 131072, grid=grid(131072), stream=stream0)
        del getitem_55
        del mm_46
        del mm_47
        del tanh_6
        buf588 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf587, (1024, 128), (1, 1024), 0), view_177, out=buf588)
        buf589 = reinterpret_tensor(buf584, (128, 512), (512, 1), 0); del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf587, (128, 1024), (1024, 1), 0), permute_811, out=buf589)
        del permute_811
        buf592 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (1024, 128), (1, 1024), 0), view_177, out=buf592)
        del view_177
        buf593 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (128, 1024), (1024, 1), 0), permute_815, out=buf593)
        del permute_815
        buf594 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_78], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf589, buf593, add_48, rsqrt_13, buf594, 512, 128, grid=grid(512), stream=stream0)
        buf596 = buf583; del buf583  # reuse
        buf597 = reinterpret_tensor(buf577, (1, 128, 512), (65536, 512, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf596, buf589, buf593, primals_14, add_48, rsqrt_13, getitem_53, buf597, 128, 512, grid=grid(128), stream=stream0)
        del add_48
        del getitem_53
        del primals_14
        del rsqrt_13
        buf598 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (512, 128), (1, 512), 0), view_175, out=buf598)
        del view_175
        buf599 = buf578; del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (128, 512), (512, 1), 0), permute_819, out=buf599)
        del permute_819
        buf600 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_822, reinterpret_tensor(buf599, (6, 128, 64), (64, 384, 1), 0), out=buf600)
        del permute_822
        buf601 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf599, (6, 128, 64), (64, 384, 1), 0), permute_823, out=buf601)
        del permute_823
        buf604 = reinterpret_tensor(buf563, (98304, ), (1, ), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf604, 98304, grid=grid(98304), stream=stream0)
        buf607 = reinterpret_tensor(buf516, (6, 128, 128), (16384, 128, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf601, getitem_51, alias_140, buf604, buf607, 768, 128, grid=grid(768), stream=stream0)
        del alias_140
        del getitem_51
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf604, buf607, 98304, grid=grid(98304), stream=stream0)
        buf609 = reinterpret_tensor(buf599, (6, 64, 128), (8192, 128, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_824, buf607, out=buf609)
        del permute_824
        buf610 = reinterpret_tensor(buf573, (6, 128, 64), (8192, 64, 1), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf607, permute_825, out=buf610)
        del permute_825
        buf611 = reinterpret_tensor(buf562, (128, 384), (384, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf600, buf611, 49152, grid=grid(49152), stream=stream0)
        buf612 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf611, (384, 128), (1, 384), 0), view_157, out=buf612)
        buf613 = reinterpret_tensor(buf597, (128, 512), (512, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf611, permute_830, out=buf613)
        del permute_830
        buf614 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf609, (384, 128), (128, 1), 0), view_157, out=buf614)
        buf615 = buf593; del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf609, (128, 384), (1, 128), 0), permute_835, out=buf615)
        del permute_835
        buf616 = reinterpret_tensor(buf609, (128, 384), (384, 1), 0); del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf610, buf616, 49152, grid=grid(49152), stream=stream0)
        buf617 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf616, (384, 128), (1, 384), 0), view_157, out=buf617)
        del view_157
        buf618 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf616, permute_840, out=buf618)
        del permute_840
        buf619 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_73], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf613, buf615, buf618, add_45, rsqrt_12, buf619, 512, 128, grid=grid(512), stream=stream0)
        buf621 = buf596; del buf596  # reuse
        buf622 = reinterpret_tensor(buf575, (1, 128, 512), (65536, 512, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf621, buf613, buf615, buf618, primals_13, add_45, rsqrt_12, getitem_49, buf622, 128, 512, grid=grid(128), stream=stream0)
        del add_45
        del getitem_49
        del primals_13
        del rsqrt_12
        buf623 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf622, (512, 128), (1, 512), 0), view_155, out=buf623)
        del view_155
        buf624 = reinterpret_tensor(buf591, (128, 1024), (1024, 1), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf622, (128, 512), (512, 1), 0), permute_844, out=buf624)
        del permute_844
        buf625 = buf587; del buf587  # reuse
        buf628 = reinterpret_tensor(buf586, (1, 128, 1024), (131072, 1024, 1), 0); del buf586  # reuse
        buf629 = buf628; del buf628  # reuse
        # Source Nodes: [add_37, hidden_gelu_5, mul_52], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf629, buf624, getitem_47, mm_39, tanh_5, mm_40, buf625, 131072, grid=grid(131072), stream=stream0)
        del getitem_47
        del mm_39
        del mm_40
        del tanh_5
        buf626 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (1024, 128), (1, 1024), 0), view_151, out=buf626)
        buf627 = reinterpret_tensor(buf622, (128, 512), (512, 1), 0); del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (128, 1024), (1024, 1), 0), permute_848, out=buf627)
        del permute_848
        buf630 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (1024, 128), (1, 1024), 0), view_151, out=buf630)
        del view_151
        buf631 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (128, 1024), (1024, 1), 0), permute_852, out=buf631)
        del permute_852
        buf632 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf627, buf631, add_41, rsqrt_11, buf632, 512, 128, grid=grid(512), stream=stream0)
        buf634 = buf621; del buf621  # reuse
        buf635 = reinterpret_tensor(buf615, (1, 128, 512), (65536, 512, 1), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf634, buf627, buf631, primals_12, add_41, rsqrt_11, getitem_45, buf635, 128, 512, grid=grid(128), stream=stream0)
        del add_41
        del getitem_45
        del primals_12
        del rsqrt_11
        buf636 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (512, 128), (1, 512), 0), view_149, out=buf636)
        del view_149
        buf637 = buf616; del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (128, 512), (512, 1), 0), permute_856, out=buf637)
        del permute_856
        buf638 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_859, reinterpret_tensor(buf637, (6, 128, 64), (64, 384, 1), 0), out=buf638)
        del permute_859
        buf639 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf637, (6, 128, 64), (64, 384, 1), 0), permute_860, out=buf639)
        del permute_860
        buf642 = reinterpret_tensor(buf601, (98304, ), (1, ), 0); del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf642, 98304, grid=grid(98304), stream=stream0)
        buf645 = reinterpret_tensor(buf450, (6, 128, 128), (16384, 128, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf639, getitem_43, alias_144, buf642, buf645, 768, 128, grid=grid(768), stream=stream0)
        del alias_144
        del getitem_43
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf642, buf645, 98304, grid=grid(98304), stream=stream0)
        buf647 = reinterpret_tensor(buf637, (6, 64, 128), (8192, 128, 1), 0); del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_861, buf645, out=buf647)
        del permute_861
        buf648 = reinterpret_tensor(buf611, (6, 128, 64), (8192, 64, 1), 0); del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf645, permute_862, out=buf648)
        del permute_862
        buf649 = reinterpret_tensor(buf600, (128, 384), (384, 1), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf638, buf649, 49152, grid=grid(49152), stream=stream0)
        buf650 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf649, (384, 128), (1, 384), 0), view_131, out=buf650)
        buf651 = reinterpret_tensor(buf635, (128, 512), (512, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf649, permute_867, out=buf651)
        del permute_867
        buf652 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf647, (384, 128), (128, 1), 0), view_131, out=buf652)
        buf653 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf647, (128, 384), (1, 128), 0), permute_872, out=buf653)
        del permute_872
        buf654 = reinterpret_tensor(buf647, (128, 384), (384, 1), 0); del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf648, buf654, 49152, grid=grid(49152), stream=stream0)
        buf655 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (384, 128), (1, 384), 0), view_131, out=buf655)
        del view_131
        buf656 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf654, permute_877, out=buf656)
        del permute_877
        buf657 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_61], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf651, buf653, buf656, add_38, rsqrt_10, buf657, 512, 128, grid=grid(512), stream=stream0)
        buf659 = buf634; del buf634  # reuse
        buf660 = reinterpret_tensor(buf613, (1, 128, 512), (65536, 512, 1), 0); del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf659, buf651, buf653, buf656, primals_11, add_38, rsqrt_10, getitem_41, buf660, 128, 512, grid=grid(128), stream=stream0)
        del add_38
        del getitem_41
        del primals_11
        del rsqrt_10
        buf661 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (512, 128), (1, 512), 0), view_129, out=buf661)
        del view_129
        buf662 = reinterpret_tensor(buf629, (128, 1024), (1024, 1), 0); del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (128, 512), (512, 1), 0), permute_881, out=buf662)
        del permute_881
        buf663 = buf625; del buf625  # reuse
        buf666 = reinterpret_tensor(buf624, (1, 128, 1024), (131072, 1024, 1), 0); del buf624  # reuse
        buf667 = buf666; del buf666  # reuse
        # Source Nodes: [add_31, hidden_gelu_4, mul_43], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf667, buf662, getitem_39, mm_32, tanh_4, mm_33, buf663, 131072, grid=grid(131072), stream=stream0)
        del getitem_39
        del mm_32
        del mm_33
        del tanh_4
        buf664 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf663, (1024, 128), (1, 1024), 0), view_125, out=buf664)
        buf665 = reinterpret_tensor(buf660, (128, 512), (512, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf663, (128, 1024), (1024, 1), 0), permute_885, out=buf665)
        del permute_885
        buf668 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (1024, 128), (1, 1024), 0), view_125, out=buf668)
        del view_125
        buf669 = buf656; del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (128, 1024), (1024, 1), 0), permute_889, out=buf669)
        del permute_889
        buf670 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_54], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf665, buf669, add_34, rsqrt_9, buf670, 512, 128, grid=grid(512), stream=stream0)
        buf672 = buf659; del buf659  # reuse
        buf673 = reinterpret_tensor(buf653, (1, 128, 512), (65536, 512, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf672, buf665, buf669, primals_10, add_34, rsqrt_9, getitem_37, buf673, 128, 512, grid=grid(128), stream=stream0)
        del add_34
        del getitem_37
        del primals_10
        del rsqrt_9
        buf674 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf673, (512, 128), (1, 512), 0), view_123, out=buf674)
        del view_123
        buf675 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf673, (128, 512), (512, 1), 0), permute_893, out=buf675)
        del permute_893
        buf676 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_896, reinterpret_tensor(buf675, (6, 128, 64), (64, 384, 1), 0), out=buf676)
        del permute_896
        buf677 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf675, (6, 128, 64), (64, 384, 1), 0), permute_897, out=buf677)
        del permute_897
        buf680 = reinterpret_tensor(buf639, (98304, ), (1, ), 0); del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf680, 98304, grid=grid(98304), stream=stream0)
        buf683 = reinterpret_tensor(buf385, (6, 128, 128), (16384, 128, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf677, getitem_35, alias_148, buf680, buf683, 768, 128, grid=grid(768), stream=stream0)
        del alias_148
        del getitem_35
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf680, buf683, 98304, grid=grid(98304), stream=stream0)
        buf685 = reinterpret_tensor(buf675, (6, 64, 128), (8192, 128, 1), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_898, buf683, out=buf685)
        del permute_898
        buf686 = reinterpret_tensor(buf649, (6, 128, 64), (8192, 64, 1), 0); del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf683, permute_899, out=buf686)
        del permute_899
        buf687 = reinterpret_tensor(buf638, (128, 384), (384, 1), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf676, buf687, 49152, grid=grid(49152), stream=stream0)
        buf688 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf687, (384, 128), (1, 384), 0), view_105, out=buf688)
        buf689 = reinterpret_tensor(buf673, (128, 512), (512, 1), 0); del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf687, permute_904, out=buf689)
        del permute_904
        buf690 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (384, 128), (128, 1), 0), view_105, out=buf690)
        buf691 = buf669; del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (128, 384), (1, 128), 0), permute_909, out=buf691)
        del permute_909
        buf692 = reinterpret_tensor(buf685, (128, 384), (384, 1), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf686, buf692, 49152, grid=grid(49152), stream=stream0)
        buf693 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf692, (384, 128), (1, 384), 0), view_105, out=buf693)
        del view_105
        buf694 = buf665; del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf692, permute_914, out=buf694)
        del permute_914
        buf695 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_49], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf689, buf691, buf694, add_31, rsqrt_8, buf695, 512, 128, grid=grid(512), stream=stream0)
        buf697 = buf672; del buf672  # reuse
        buf698 = reinterpret_tensor(buf651, (1, 128, 512), (65536, 512, 1), 0); del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf697, buf689, buf691, buf694, primals_9, add_31, rsqrt_8, getitem_33, buf698, 128, 512, grid=grid(128), stream=stream0)
        del add_31
        del getitem_33
        del primals_9
        del rsqrt_8
        buf699 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf698, (512, 128), (1, 512), 0), view_103, out=buf699)
        del view_103
        buf700 = reinterpret_tensor(buf667, (128, 1024), (1024, 1), 0); del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf698, (128, 512), (512, 1), 0), permute_918, out=buf700)
        del permute_918
        buf701 = buf663; del buf663  # reuse
        buf704 = reinterpret_tensor(buf662, (1, 128, 1024), (131072, 1024, 1), 0); del buf662  # reuse
        buf705 = buf704; del buf704  # reuse
        # Source Nodes: [add_25, hidden_gelu_3, mul_34], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf705, buf700, getitem_31, mm_25, tanh_3, mm_26, buf701, 131072, grid=grid(131072), stream=stream0)
        del getitem_31
        del mm_25
        del mm_26
        del tanh_3
        buf702 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (1024, 128), (1, 1024), 0), view_99, out=buf702)
        buf703 = reinterpret_tensor(buf698, (128, 512), (512, 1), 0); del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (128, 1024), (1024, 1), 0), permute_922, out=buf703)
        del permute_922
        buf706 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (1024, 128), (1, 1024), 0), view_99, out=buf706)
        del view_99
        buf707 = buf694; del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (128, 1024), (1024, 1), 0), permute_926, out=buf707)
        del permute_926
        buf708 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_42], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf703, buf707, add_27, rsqrt_7, buf708, 512, 128, grid=grid(512), stream=stream0)
        buf710 = buf697; del buf697  # reuse
        buf711 = reinterpret_tensor(buf691, (1, 128, 512), (65536, 512, 1), 0); del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf710, buf703, buf707, primals_8, add_27, rsqrt_7, getitem_29, buf711, 128, 512, grid=grid(128), stream=stream0)
        del add_27
        del getitem_29
        del primals_8
        del rsqrt_7
        buf712 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf711, (512, 128), (1, 512), 0), view_97, out=buf712)
        del view_97
        buf713 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf711, (128, 512), (512, 1), 0), permute_930, out=buf713)
        del permute_930
        buf714 = buf686; del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_933, reinterpret_tensor(buf713, (6, 128, 64), (64, 384, 1), 0), out=buf714)
        del permute_933
        buf715 = buf683; del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf713, (6, 128, 64), (64, 384, 1), 0), permute_934, out=buf715)
        del permute_934
        buf718 = reinterpret_tensor(buf677, (98304, ), (1, ), 0); del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf718, 98304, grid=grid(98304), stream=stream0)
        buf721 = reinterpret_tensor(buf320, (6, 128, 128), (16384, 128, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf715, getitem_27, alias_152, buf718, buf721, 768, 128, grid=grid(768), stream=stream0)
        del alias_152
        del getitem_27
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf718, buf721, 98304, grid=grid(98304), stream=stream0)
        buf723 = reinterpret_tensor(buf713, (6, 64, 128), (8192, 128, 1), 0); del buf713  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_935, buf721, out=buf723)
        del permute_935
        buf724 = reinterpret_tensor(buf687, (6, 128, 64), (8192, 64, 1), 0); del buf687  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf721, permute_936, out=buf724)
        del permute_936
        buf725 = reinterpret_tensor(buf676, (128, 384), (384, 1), 0); del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf714, buf725, 49152, grid=grid(49152), stream=stream0)
        buf726 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (384, 128), (1, 384), 0), view_79, out=buf726)
        buf727 = reinterpret_tensor(buf711, (128, 512), (512, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf725, permute_941, out=buf727)
        del permute_941
        buf728 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf723, (384, 128), (128, 1), 0), view_79, out=buf728)
        buf729 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf723, (128, 384), (1, 128), 0), permute_946, out=buf729)
        del permute_946
        buf730 = reinterpret_tensor(buf723, (128, 384), (384, 1), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf724, buf730, 49152, grid=grid(49152), stream=stream0)
        buf731 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf730, (384, 128), (1, 384), 0), view_79, out=buf731)
        del view_79
        buf732 = buf703; del buf703  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf730, permute_951, out=buf732)
        del permute_951
        buf733 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_37], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf727, buf729, buf732, add_24, rsqrt_6, buf733, 512, 128, grid=grid(512), stream=stream0)
        buf735 = buf710; del buf710  # reuse
        buf736 = reinterpret_tensor(buf689, (1, 128, 512), (65536, 512, 1), 0); del buf689  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf735, buf727, buf729, buf732, primals_7, add_24, rsqrt_6, getitem_25, buf736, 128, 512, grid=grid(128), stream=stream0)
        del add_24
        del getitem_25
        del primals_7
        del rsqrt_6
        buf737 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (512, 128), (1, 512), 0), view_77, out=buf737)
        del view_77
        buf738 = reinterpret_tensor(buf705, (128, 1024), (1024, 1), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (128, 512), (512, 1), 0), permute_955, out=buf738)
        del permute_955
        buf739 = buf701; del buf701  # reuse
        buf742 = reinterpret_tensor(buf700, (1, 128, 1024), (131072, 1024, 1), 0); del buf700  # reuse
        buf743 = buf742; del buf742  # reuse
        # Source Nodes: [add_19, hidden_gelu_2, mul_25], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf743, buf738, getitem_23, mm_18, tanh_2, mm_19, buf739, 131072, grid=grid(131072), stream=stream0)
        del getitem_23
        del mm_18
        del mm_19
        del tanh_2
        buf740 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf739, (1024, 128), (1, 1024), 0), view_73, out=buf740)
        buf741 = reinterpret_tensor(buf736, (128, 512), (512, 1), 0); del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf739, (128, 1024), (1024, 1), 0), permute_959, out=buf741)
        del permute_959
        buf744 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (1024, 128), (1, 1024), 0), view_73, out=buf744)
        del view_73
        buf745 = buf732; del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (128, 1024), (1024, 1), 0), permute_963, out=buf745)
        del permute_963
        buf746 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_30], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf741, buf745, add_20, rsqrt_5, buf746, 512, 128, grid=grid(512), stream=stream0)
        buf748 = buf735; del buf735  # reuse
        buf749 = reinterpret_tensor(buf729, (1, 128, 512), (65536, 512, 1), 0); del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf748, buf741, buf745, primals_6, add_20, rsqrt_5, getitem_21, buf749, 128, 512, grid=grid(128), stream=stream0)
        del add_20
        del getitem_21
        del primals_6
        del rsqrt_5
        buf750 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (512, 128), (1, 512), 0), view_71, out=buf750)
        del view_71
        buf751 = buf730; del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (128, 512), (512, 1), 0), permute_967, out=buf751)
        del permute_967
        buf752 = buf724; del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_970, reinterpret_tensor(buf751, (6, 128, 64), (64, 384, 1), 0), out=buf752)
        del permute_970
        buf753 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf751, (6, 128, 64), (64, 384, 1), 0), permute_971, out=buf753)
        del permute_971
        buf756 = reinterpret_tensor(buf715, (98304, ), (1, ), 0); del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf756, 98304, grid=grid(98304), stream=stream0)
        buf759 = reinterpret_tensor(buf255, (6, 128, 128), (16384, 128, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf753, getitem_19, alias_156, buf756, buf759, 768, 128, grid=grid(768), stream=stream0)
        del alias_156
        del getitem_19
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf756, buf759, 98304, grid=grid(98304), stream=stream0)
        buf761 = reinterpret_tensor(buf751, (6, 64, 128), (8192, 128, 1), 0); del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_972, buf759, out=buf761)
        del permute_972
        buf762 = reinterpret_tensor(buf725, (6, 128, 64), (8192, 64, 1), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf759, permute_973, out=buf762)
        del permute_973
        buf763 = reinterpret_tensor(buf714, (128, 384), (384, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf752, buf763, 49152, grid=grid(49152), stream=stream0)
        buf764 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf763, (384, 128), (1, 384), 0), view_53, out=buf764)
        buf765 = reinterpret_tensor(buf749, (128, 512), (512, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf763, permute_978, out=buf765)
        del permute_978
        buf766 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (384, 128), (128, 1), 0), view_53, out=buf766)
        buf767 = buf745; del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (128, 384), (1, 128), 0), permute_983, out=buf767)
        del permute_983
        buf768 = reinterpret_tensor(buf761, (128, 384), (384, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf762, buf768, 49152, grid=grid(49152), stream=stream0)
        buf769 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf768, (384, 128), (1, 384), 0), view_53, out=buf769)
        del view_53
        buf770 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf768, permute_988, out=buf770)
        del permute_988
        buf771 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_25], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf765, buf767, buf770, add_17, rsqrt_4, buf771, 512, 128, grid=grid(512), stream=stream0)
        buf773 = buf748; del buf748  # reuse
        buf774 = reinterpret_tensor(buf727, (1, 128, 512), (65536, 512, 1), 0); del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf773, buf765, buf767, buf770, primals_5, add_17, rsqrt_4, getitem_17, buf774, 128, 512, grid=grid(128), stream=stream0)
        del add_17
        del getitem_17
        del primals_5
        del rsqrt_4
        buf775 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf774, (512, 128), (1, 512), 0), view_51, out=buf775)
        del view_51
        buf776 = reinterpret_tensor(buf743, (128, 1024), (1024, 1), 0); del buf743  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf774, (128, 512), (512, 1), 0), permute_992, out=buf776)
        del permute_992
        buf777 = buf739; del buf739  # reuse
        buf780 = reinterpret_tensor(buf738, (1, 128, 1024), (131072, 1024, 1), 0); del buf738  # reuse
        buf781 = buf780; del buf780  # reuse
        # Source Nodes: [add_13, hidden_gelu_1, mul_16], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf781, buf776, getitem_15, mm_11, tanh_1, mm_12, buf777, 131072, grid=grid(131072), stream=stream0)
        del getitem_15
        del mm_11
        del mm_12
        del tanh_1
        buf778 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (1024, 128), (1, 1024), 0), view_47, out=buf778)
        buf779 = reinterpret_tensor(buf774, (128, 512), (512, 1), 0); del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (128, 1024), (1024, 1), 0), permute_996, out=buf779)
        del permute_996
        buf782 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf781, (1024, 128), (1, 1024), 0), view_47, out=buf782)
        del view_47
        buf783 = buf770; del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf781, (128, 1024), (1024, 1), 0), permute_1000, out=buf783)
        del permute_1000
        buf784 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf779, buf783, add_13, rsqrt_3, buf784, 512, 128, grid=grid(512), stream=stream0)
        buf786 = buf773; del buf773  # reuse
        buf787 = reinterpret_tensor(buf767, (1, 128, 512), (65536, 512, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf786, buf779, buf783, primals_4, add_13, rsqrt_3, getitem_13, buf787, 128, 512, grid=grid(128), stream=stream0)
        del add_13
        del getitem_13
        del primals_4
        del rsqrt_3
        buf788 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf787, (512, 128), (1, 512), 0), view_45, out=buf788)
        del view_45
        buf789 = buf768; del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf787, (128, 512), (512, 1), 0), permute_1004, out=buf789)
        del permute_1004
        buf790 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1007, reinterpret_tensor(buf789, (6, 128, 64), (64, 384, 1), 0), out=buf790)
        del permute_1007
        buf791 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf789, (6, 128, 64), (64, 384, 1), 0), permute_1008, out=buf791)
        del permute_1008
        buf794 = reinterpret_tensor(buf753, (98304, ), (1, ), 0); del buf753  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf794, 98304, grid=grid(98304), stream=stream0)
        buf797 = reinterpret_tensor(buf189, (6, 128, 128), (16384, 128, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf791, getitem_11, alias_160, buf794, buf797, 768, 128, grid=grid(768), stream=stream0)
        del alias_160
        del getitem_11
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_12.run(buf794, buf797, 98304, grid=grid(98304), stream=stream0)
        buf799 = reinterpret_tensor(buf789, (6, 64, 128), (8192, 128, 1), 0); del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1009, buf797, out=buf799)
        del permute_1009
        buf800 = reinterpret_tensor(buf763, (6, 128, 64), (8192, 64, 1), 0); del buf763  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf797, permute_1010, out=buf800)
        del permute_1010
        buf801 = reinterpret_tensor(buf752, (128, 384), (384, 1), 0); del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf790, buf801, 49152, grid=grid(49152), stream=stream0)
        buf802 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (384, 128), (1, 384), 0), view_27, out=buf802)
        buf803 = reinterpret_tensor(buf787, (128, 512), (512, 1), 0); del buf787  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf801, permute_1015, out=buf803)
        del permute_1015
        buf804 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf799, (384, 128), (128, 1), 0), view_27, out=buf804)
        buf805 = buf783; del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf799, (128, 384), (1, 128), 0), permute_1020, out=buf805)
        del permute_1020
        buf806 = reinterpret_tensor(buf799, (128, 384), (384, 1), 0); del buf799  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf800, buf806, 49152, grid=grid(49152), stream=stream0)
        buf807 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (384, 128), (1, 384), 0), view_27, out=buf807)
        del view_27
        buf808 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf806, permute_1025, out=buf808)
        del permute_1025
        buf809 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf803, buf805, buf808, add_10, rsqrt_2, buf809, 512, 128, grid=grid(512), stream=stream0)
        buf811 = buf786; del buf786  # reuse
        buf812 = reinterpret_tensor(buf765, (1, 128, 512), (65536, 512, 1), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_19.run(buf811, buf803, buf805, buf808, primals_3, add_10, rsqrt_2, getitem_9, buf812, 128, 512, grid=grid(128), stream=stream0)
        del add_10
        del buf803
        del getitem_9
        del primals_3
        del rsqrt_2
        buf813 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (512, 128), (1, 512), 0), view_25, out=buf813)
        del view_25
        buf814 = reinterpret_tensor(buf781, (128, 1024), (1024, 1), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (128, 512), (512, 1), 0), permute_1029, out=buf814)
        del permute_1029
        buf815 = buf777; del buf777  # reuse
        buf818 = reinterpret_tensor(buf776, (1, 128, 1024), (131072, 1024, 1), 0); del buf776  # reuse
        buf819 = buf818; del buf818  # reuse
        # Source Nodes: [add_7, hidden_gelu, mul_7], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7.run(buf819, buf814, getitem_7, mm_4, tanh, mm_5, buf815, 131072, grid=grid(131072), stream=stream0)
        del buf814
        del getitem_7
        del mm_4
        del mm_5
        del tanh
        buf816 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf815, (1024, 128), (1, 1024), 0), view_21, out=buf816)
        buf817 = reinterpret_tensor(buf812, (128, 512), (512, 1), 0); del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf815, (128, 1024), (1024, 1), 0), permute_1033, out=buf817)
        del buf815
        del permute_1033
        buf820 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (1024, 128), (1, 1024), 0), view_21, out=buf820)
        del view_21
        buf821 = buf808; del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (128, 1024), (1024, 1), 0), permute_1037, out=buf821)
        del buf819
        del permute_1037
        buf822 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_6], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_8.run(buf817, buf821, add_6, rsqrt_1, buf822, 512, 128, grid=grid(512), stream=stream0)
        buf824 = buf811; del buf811  # reuse
        buf825 = reinterpret_tensor(buf805, (1, 128, 512), (65536, 512, 1), 0); del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_9.run(buf824, buf817, buf821, primals_2, add_6, rsqrt_1, getitem_5, buf825, 128, 512, grid=grid(128), stream=stream0)
        del add_6
        del getitem_5
        del primals_2
        del rsqrt_1
        buf826 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf825, (512, 128), (1, 512), 0), view_19, out=buf826)
        del view_19
        buf827 = buf806; del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf825, (128, 512), (512, 1), 0), permute_1041, out=buf827)
        del permute_1041
        buf828 = buf800; del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1044, reinterpret_tensor(buf827, (6, 128, 64), (64, 384, 1), 0), out=buf828)
        del permute_1044
        buf829 = buf797; del buf797  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf827, (6, 128, 64), (64, 384, 1), 0), permute_1045, out=buf829)
        del permute_1045
        buf832 = reinterpret_tensor(buf791, (98304, ), (1, ), 0); del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf832, 98304, grid=grid(98304), stream=stream0)
        buf835 = reinterpret_tensor(buf124, (6, 128, 128), (16384, 128, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_11.run(buf829, getitem_3, alias_164, buf832, buf835, 768, 128, grid=grid(768), stream=stream0)
        del alias_164
        del getitem_3
        buf838 = reinterpret_tensor(buf829, (128, 128, 6), (128, 1, 16384), 0); del buf829  # reuse
        # Source Nodes: [loss], Original ATen: [aten.as_strided_scatter, aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_21.run(buf832, buf566, buf604, buf642, buf680, buf718, buf756, buf794, buf835, buf838, 98304, grid=grid(98304), stream=stream0)
        del buf566
        del buf604
        del buf642
        del buf680
        del buf718
        del buf756
        del buf794
        del buf832
        buf837 = empty((32, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_22.run(buf837, 192, grid=grid(192), stream=stream0)
        aten.index_put_(buf837, [add_3], buf838, True)
        del add_3
        del buf838
        buf841 = reinterpret_tensor(buf827, (6, 64, 128), (8192, 128, 1), 0); del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1047, buf835, out=buf841)
        del permute_1047
        buf842 = reinterpret_tensor(buf801, (6, 128, 64), (8192, 64, 1), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf835, permute_1048, out=buf842)
        del buf835
        del permute_1048
        buf843 = reinterpret_tensor(buf790, (128, 384), (384, 1), 0); del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf828, buf843, 49152, grid=grid(49152), stream=stream0)
        del buf828
        buf844 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf843, (384, 128), (1, 384), 0), view_1, out=buf844)
        buf845 = reinterpret_tensor(buf825, (128, 512), (512, 1), 0); del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf843, permute_1053, out=buf845)
        del buf843
        del permute_1053
        buf846 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf841, (384, 128), (128, 1), 0), view_1, out=buf846)
        buf847 = buf821; del buf821  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf841, (128, 384), (1, 128), 0), permute_1058, out=buf847)
        del permute_1058
        buf848 = reinterpret_tensor(buf841, (128, 384), (384, 1), 0); del buf841  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf842, buf848, 49152, grid=grid(49152), stream=stream0)
        del buf842
        buf849 = empty((384, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf848, (384, 128), (1, 384), 0), view_1, out=buf849)
        del view_1
        buf850 = buf817; del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf848, permute_1063, out=buf850)
        del buf848
        del permute_1063
        buf851 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_18.run(buf845, buf847, buf850, getitem, rsqrt, buf851, 512, 128, grid=grid(512), stream=stream0)
        buf853 = buf824; del buf824  # reuse
        buf855 = buf853; del buf853  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum]
        triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_23.run(buf855, buf845, buf847, buf850, primals_1, getitem, rsqrt, getitem_1, view, 128, 512, grid=grid(128), stream=stream0)
        del buf845
        del buf847
        del buf850
        del getitem
        del getitem_1
        del primals_1
        del rsqrt
        buf854 = empty((250112, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_24.run(buf854, 128057344, grid=grid(128057344), stream=stream0)
        aten.index_put_(buf854, [view], buf855, True)
        del buf855
        del view
        buf542 = empty((250112, 512), device='cuda', dtype=torch.float32)
        buf858 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_25.run(buf858, buf539, buf854, 128057344, grid=grid(128057344), stream=stream0)
        return (reinterpret_tensor(buf851, (512, ), (1, ), 0), reinterpret_tensor(buf822, (512, ), (1, ), 0), reinterpret_tensor(buf809, (512, ), (1, ), 0), reinterpret_tensor(buf784, (512, ), (1, ), 0), reinterpret_tensor(buf771, (512, ), (1, ), 0), reinterpret_tensor(buf746, (512, ), (1, ), 0), reinterpret_tensor(buf733, (512, ), (1, ), 0), reinterpret_tensor(buf708, (512, ), (1, ), 0), reinterpret_tensor(buf695, (512, ), (1, ), 0), reinterpret_tensor(buf670, (512, ), (1, ), 0), reinterpret_tensor(buf657, (512, ), (1, ), 0), reinterpret_tensor(buf632, (512, ), (1, ), 0), reinterpret_tensor(buf619, (512, ), (1, ), 0), reinterpret_tensor(buf594, (512, ), (1, ), 0), reinterpret_tensor(buf581, (512, ), (1, ), 0), reinterpret_tensor(buf556, (512, ), (1, ), 0), reinterpret_tensor(buf543, (512, ), (1, ), 0), reinterpret_tensor(buf536, (512, ), (1, ), 0), reinterpret_tensor(buf506, (512, ), (1, ), 0), reinterpret_tensor(buf479, (512, ), (1, ), 0), reinterpret_tensor(buf466, (512, ), (1, ), 0), reinterpret_tensor(buf440, (512, ), (1, ), 0), reinterpret_tensor(buf414, (512, ), (1, ), 0), reinterpret_tensor(buf401, (512, ), (1, ), 0), reinterpret_tensor(buf375, (512, ), (1, ), 0), reinterpret_tensor(buf349, (512, ), (1, ), 0), reinterpret_tensor(buf336, (512, ), (1, ), 0), reinterpret_tensor(buf310, (512, ), (1, ), 0), reinterpret_tensor(buf284, (512, ), (1, ), 0), reinterpret_tensor(buf271, (512, ), (1, ), 0), reinterpret_tensor(buf245, (512, ), (1, ), 0), reinterpret_tensor(buf218, (512, ), (1, ), 0), reinterpret_tensor(buf205, (512, ), (1, ), 0), reinterpret_tensor(buf179, (512, ), (1, ), 0), reinterpret_tensor(buf153, (512, ), (1, ), 0), reinterpret_tensor(buf140, (512, ), (1, ), 0), reinterpret_tensor(buf114, (512, ), (1, ), 0), reinterpret_tensor(buf88, (512, ), (1, ), 0), reinterpret_tensor(buf75, (512, ), (1, ), 0), reinterpret_tensor(buf49, (512, ), (1, ), 0), reinterpret_tensor(buf22, (512, ), (1, ), 0), reinterpret_tensor(buf9, (512, ), (1, ), 0), buf858, reinterpret_tensor(buf849, (384, 512), (512, 1), 0), reinterpret_tensor(buf846, (384, 512), (512, 1), 0), reinterpret_tensor(buf844, (384, 512), (512, 1), 0), buf837, reinterpret_tensor(buf826, (512, 384), (384, 1), 0), reinterpret_tensor(buf820, (1024, 512), (512, 1), 0), reinterpret_tensor(buf816, (1024, 512), (512, 1), 0), reinterpret_tensor(buf813, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf807, (384, 512), (512, 1), 0), reinterpret_tensor(buf804, (384, 512), (512, 1), 0), reinterpret_tensor(buf802, (384, 512), (512, 1), 0), reinterpret_tensor(buf788, (512, 384), (384, 1), 0), reinterpret_tensor(buf782, (1024, 512), (512, 1), 0), reinterpret_tensor(buf778, (1024, 512), (512, 1), 0), reinterpret_tensor(buf775, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf769, (384, 512), (512, 1), 0), reinterpret_tensor(buf766, (384, 512), (512, 1), 0), reinterpret_tensor(buf764, (384, 512), (512, 1), 0), reinterpret_tensor(buf750, (512, 384), (384, 1), 0), reinterpret_tensor(buf744, (1024, 512), (512, 1), 0), reinterpret_tensor(buf740, (1024, 512), (512, 1), 0), reinterpret_tensor(buf737, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf731, (384, 512), (512, 1), 0), reinterpret_tensor(buf728, (384, 512), (512, 1), 0), reinterpret_tensor(buf726, (384, 512), (512, 1), 0), reinterpret_tensor(buf712, (512, 384), (384, 1), 0), reinterpret_tensor(buf706, (1024, 512), (512, 1), 0), reinterpret_tensor(buf702, (1024, 512), (512, 1), 0), reinterpret_tensor(buf699, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf693, (384, 512), (512, 1), 0), reinterpret_tensor(buf690, (384, 512), (512, 1), 0), reinterpret_tensor(buf688, (384, 512), (512, 1), 0), reinterpret_tensor(buf674, (512, 384), (384, 1), 0), reinterpret_tensor(buf668, (1024, 512), (512, 1), 0), reinterpret_tensor(buf664, (1024, 512), (512, 1), 0), reinterpret_tensor(buf661, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf655, (384, 512), (512, 1), 0), reinterpret_tensor(buf652, (384, 512), (512, 1), 0), reinterpret_tensor(buf650, (384, 512), (512, 1), 0), reinterpret_tensor(buf636, (512, 384), (384, 1), 0), reinterpret_tensor(buf630, (1024, 512), (512, 1), 0), reinterpret_tensor(buf626, (1024, 512), (512, 1), 0), reinterpret_tensor(buf623, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf617, (384, 512), (512, 1), 0), reinterpret_tensor(buf614, (384, 512), (512, 1), 0), reinterpret_tensor(buf612, (384, 512), (512, 1), 0), reinterpret_tensor(buf598, (512, 384), (384, 1), 0), reinterpret_tensor(buf592, (1024, 512), (512, 1), 0), reinterpret_tensor(buf588, (1024, 512), (512, 1), 0), reinterpret_tensor(buf585, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf579, (384, 512), (512, 1), 0), reinterpret_tensor(buf576, (384, 512), (512, 1), 0), reinterpret_tensor(buf574, (384, 512), (512, 1), 0), reinterpret_tensor(buf560, (512, 384), (384, 1), 0), reinterpret_tensor(buf554, (1024, 512), (512, 1), 0), reinterpret_tensor(buf550, (1024, 512), (512, 1), 0), reinterpret_tensor(buf547, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf534, (384, 512), (512, 1), 0), reinterpret_tensor(buf531, (384, 512), (512, 1), 0), reinterpret_tensor(buf528, (384, 512), (512, 1), 0), buf521, reinterpret_tensor(buf510, (512, 384), (384, 1), 0), reinterpret_tensor(buf504, (384, 512), (512, 1), 0), reinterpret_tensor(buf500, (384, 512), (512, 1), 0), reinterpret_tensor(buf497, (384, 512), (512, 1), 0), reinterpret_tensor(buf483, (512, 384), (384, 1), 0), reinterpret_tensor(buf477, (1024, 512), (512, 1), 0), reinterpret_tensor(buf473, (1024, 512), (512, 1), 0), reinterpret_tensor(buf470, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf464, (384, 512), (512, 1), 0), reinterpret_tensor(buf461, (384, 512), (512, 1), 0), reinterpret_tensor(buf458, (384, 512), (512, 1), 0), reinterpret_tensor(buf444, (512, 384), (384, 1), 0), reinterpret_tensor(buf438, (384, 512), (512, 1), 0), reinterpret_tensor(buf435, (384, 512), (512, 1), 0), reinterpret_tensor(buf432, (384, 512), (512, 1), 0), reinterpret_tensor(buf418, (512, 384), (384, 1), 0), reinterpret_tensor(buf412, (1024, 512), (512, 1), 0), reinterpret_tensor(buf408, (1024, 512), (512, 1), 0), reinterpret_tensor(buf405, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf399, (384, 512), (512, 1), 0), reinterpret_tensor(buf396, (384, 512), (512, 1), 0), reinterpret_tensor(buf393, (384, 512), (512, 1), 0), reinterpret_tensor(buf379, (512, 384), (384, 1), 0), reinterpret_tensor(buf373, (384, 512), (512, 1), 0), reinterpret_tensor(buf370, (384, 512), (512, 1), 0), reinterpret_tensor(buf367, (384, 512), (512, 1), 0), reinterpret_tensor(buf353, (512, 384), (384, 1), 0), reinterpret_tensor(buf347, (1024, 512), (512, 1), 0), reinterpret_tensor(buf343, (1024, 512), (512, 1), 0), reinterpret_tensor(buf340, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf334, (384, 512), (512, 1), 0), reinterpret_tensor(buf331, (384, 512), (512, 1), 0), reinterpret_tensor(buf328, (384, 512), (512, 1), 0), reinterpret_tensor(buf314, (512, 384), (384, 1), 0), reinterpret_tensor(buf308, (384, 512), (512, 1), 0), reinterpret_tensor(buf305, (384, 512), (512, 1), 0), reinterpret_tensor(buf302, (384, 512), (512, 1), 0), reinterpret_tensor(buf288, (512, 384), (384, 1), 0), reinterpret_tensor(buf282, (1024, 512), (512, 1), 0), reinterpret_tensor(buf278, (1024, 512), (512, 1), 0), reinterpret_tensor(buf275, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf269, (384, 512), (512, 1), 0), reinterpret_tensor(buf266, (384, 512), (512, 1), 0), reinterpret_tensor(buf263, (384, 512), (512, 1), 0), reinterpret_tensor(buf249, (512, 384), (384, 1), 0), reinterpret_tensor(buf243, (384, 512), (512, 1), 0), reinterpret_tensor(buf239, (384, 512), (512, 1), 0), reinterpret_tensor(buf236, (384, 512), (512, 1), 0), reinterpret_tensor(buf222, (512, 384), (384, 1), 0), reinterpret_tensor(buf216, (1024, 512), (512, 1), 0), reinterpret_tensor(buf212, (1024, 512), (512, 1), 0), reinterpret_tensor(buf209, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf203, (384, 512), (512, 1), 0), reinterpret_tensor(buf200, (384, 512), (512, 1), 0), reinterpret_tensor(buf197, (384, 512), (512, 1), 0), reinterpret_tensor(buf183, (512, 384), (384, 1), 0), reinterpret_tensor(buf177, (384, 512), (512, 1), 0), reinterpret_tensor(buf174, (384, 512), (512, 1), 0), reinterpret_tensor(buf171, (384, 512), (512, 1), 0), reinterpret_tensor(buf157, (512, 384), (384, 1), 0), reinterpret_tensor(buf151, (1024, 512), (512, 1), 0), reinterpret_tensor(buf147, (1024, 512), (512, 1), 0), reinterpret_tensor(buf144, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf138, (384, 512), (512, 1), 0), reinterpret_tensor(buf135, (384, 512), (512, 1), 0), reinterpret_tensor(buf132, (384, 512), (512, 1), 0), reinterpret_tensor(buf118, (512, 384), (384, 1), 0), reinterpret_tensor(buf112, (384, 512), (512, 1), 0), reinterpret_tensor(buf109, (384, 512), (512, 1), 0), reinterpret_tensor(buf106, (384, 512), (512, 1), 0), reinterpret_tensor(buf92, (512, 384), (384, 1), 0), reinterpret_tensor(buf86, (1024, 512), (512, 1), 0), reinterpret_tensor(buf82, (1024, 512), (512, 1), 0), reinterpret_tensor(buf79, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf73, (384, 512), (512, 1), 0), reinterpret_tensor(buf70, (384, 512), (512, 1), 0), reinterpret_tensor(buf67, (384, 512), (512, 1), 0), reinterpret_tensor(buf53, (512, 384), (384, 1), 0), reinterpret_tensor(buf47, (384, 512), (512, 1), 0), reinterpret_tensor(buf44, (384, 512), (512, 1), 0), reinterpret_tensor(buf41, (384, 512), (512, 1), 0), reinterpret_tensor(buf26, (512, 384), (384, 1), 0), reinterpret_tensor(buf20, (1024, 512), (512, 1), 0), reinterpret_tensor(buf16, (1024, 512), (512, 1), 0), reinterpret_tensor(buf13, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf7, (250112, 512), (512, 1), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    getitem = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    rsqrt = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_3 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    getitem_3 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_19 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_6 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_25 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_10 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_45 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_13 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_11 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_12 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_51 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_17 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_4 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_71 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_20 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_18 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_19 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_23 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_77 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_24 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_97 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_27 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_25 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_26 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_103 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_33 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_31 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_8 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_123 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_34 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_32 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_33 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_129 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_38 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_149 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_45 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_41 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_151 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_39 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_40 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_155 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_45 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_12 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_157 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_175 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_48 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_46 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_6 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_47 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_55 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_181 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_52 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_14 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_183 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_201 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_55 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_203 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_53 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_7 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_54 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_63 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_207 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_65 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_59 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    view_209 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    getitem_68 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    rsqrt_17 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_210 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_63 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    getitem_71 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_228 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_66 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_233 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_75 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_248 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_70 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_64 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_8 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_65 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_254 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_74 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_256 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_274 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_77 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_21 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_276 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_294 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_89 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_80 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_22 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_296 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_75 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_9 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_76 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_300 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_93 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_84 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_23 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_302 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_320 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_87 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_322 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_99 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_340 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_90 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_342 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_86 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_10 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_87 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_346 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_94 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_26 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_348 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_366 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_97 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_368 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_386 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_100 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_388 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_97 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_11 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_98 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_115 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_392 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_104 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_29 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_394 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_119 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_412 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_107 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_414 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_432 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_110 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_434 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_108 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_12 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_109 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_438 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_129 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_114 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_32 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_440 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_458 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_133 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_117 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_460 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_478 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_137 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_120 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_34 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_480 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_119 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_13 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_120 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_139 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_484 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_124 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_35 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_486 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_143 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_504 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_145 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_127 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_36 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_506 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_147 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_524 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_130 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_37 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_526 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_130 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_14 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_131 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_530 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_153 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_134 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_38 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_532 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_550 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_137 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_39 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_552 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_159 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_570 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_161 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_140 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_40 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_572 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_141 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tanh_15 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    mm_142 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_576 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_165 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    add_144 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_41 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_167 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    view_578 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    sub_30 = rand_strided((128, 250112), (250112, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_7 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((250112, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_281 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_87 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_296 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_301 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_314 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_89 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_326 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_331 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_339 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_347 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_351 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_93 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_353 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_358 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_363 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_372 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_95 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_397 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_401 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_409 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_412 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_99 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_437 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_101 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_445 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_474 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_105 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_477 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_499 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_107 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_502 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_507 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_512 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_529 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_533 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_536 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_537 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_111 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_538 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_539 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_544 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_549 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_554 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_561 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_562 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_113 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_563 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_564 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_569 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_574 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_579 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_583 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_587 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_595 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_598 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_599 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_117 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_600 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_601 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_606 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_611 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_616 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_620 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_623 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_119 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_626 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_631 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_636 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_641 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_645 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_649 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_653 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_657 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_660 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_661 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_123 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_662 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_663 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_668 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_673 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_678 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_682 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_685 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_686 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_125 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_688 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_693 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_698 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_703 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_707 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_711 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_715 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_719 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_722 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_723 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_129 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_724 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_725 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_730 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_735 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_740 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_744 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_747 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_748 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_131 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_750 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_751 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_756 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_761 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_766 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_770 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_774 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_778 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_782 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_785 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_786 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_136 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_787 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_788 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_793 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_798 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_803 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_807 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_811 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_815 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_819 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_823 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_140 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_824 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_825 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_830 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_835 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_840 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_844 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_848 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_852 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_856 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_859 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_860 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_144 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_861 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_862 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_867 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_872 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_877 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_881 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_885 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_889 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_893 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_896 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_897 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_148 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_898 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_899 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_904 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_909 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_914 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_918 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_922 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_926 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_930 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_933 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_934 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_152 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_935 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_936 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_941 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_946 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_951 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_955 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_959 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_963 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_967 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_970 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_971 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_156 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_972 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_973 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_978 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_983 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_988 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_992 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_996 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1000 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1004 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_1007 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1008 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_160 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_1009 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_1010 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_1015 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1020 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1025 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1029 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1033 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1037 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1041 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_1044 = rand_strided((6, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_1045 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    alias_164 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_1047 = rand_strided((6, 64, 128), (64, 1, 384), device='cuda:0', dtype=torch.float32)
    permute_1048 = rand_strided((6, 128, 64), (64, 384, 1), device='cuda:0', dtype=torch.float32)
    permute_1053 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1058 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1063 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 250112), (32014336, 250112, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_28 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_31 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_34 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_192, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, mm_4, tanh, mm_5, getitem_7, view_25, getitem_9, add_10, rsqrt_2, view_27, getitem_11, view_45, getitem_13, add_13, rsqrt_3, view_47, mm_11, tanh_1, mm_12, getitem_15, view_51, getitem_17, add_17, rsqrt_4, view_53, getitem_19, view_71, getitem_21, add_20, rsqrt_5, view_73, mm_18, tanh_2, mm_19, getitem_23, view_77, getitem_25, add_24, rsqrt_6, view_79, getitem_27, view_97, getitem_29, add_27, rsqrt_7, view_99, mm_25, tanh_3, mm_26, getitem_31, view_103, getitem_33, add_31, rsqrt_8, view_105, getitem_35, view_123, getitem_37, add_34, rsqrt_9, view_125, mm_32, tanh_4, mm_33, getitem_39, view_129, getitem_41, add_38, rsqrt_10, view_131, getitem_43, view_149, getitem_45, add_41, rsqrt_11, view_151, mm_39, tanh_5, mm_40, getitem_47, view_155, getitem_49, add_45, rsqrt_12, view_157, getitem_51, view_175, getitem_53, add_48, rsqrt_13, view_177, mm_46, tanh_6, mm_47, getitem_55, view_181, getitem_57, add_52, rsqrt_14, view_183, getitem_59, view_201, getitem_61, add_55, rsqrt_15, view_203, mm_53, tanh_7, mm_54, getitem_63, view_207, getitem_65, add_59, rsqrt_16, getitem_67, view_209, getitem_68, getitem_69, rsqrt_17, view_210, add_63, getitem_71, view_228, getitem_73, add_66, rsqrt_18, view_230, view_233, getitem_75, view_248, getitem_77, add_70, rsqrt_19, view_250, mm_64, tanh_8, mm_65, getitem_79, view_254, getitem_81, add_74, rsqrt_20, view_256, getitem_83, view_274, getitem_85, add_77, rsqrt_21, view_276, getitem_87, view_294, getitem_89, add_80, rsqrt_22, view_296, mm_75, tanh_9, mm_76, getitem_91, view_300, getitem_93, add_84, rsqrt_23, view_302, getitem_95, view_320, getitem_97, add_87, rsqrt_24, view_322, getitem_99, view_340, getitem_101, add_90, rsqrt_25, view_342, mm_86, tanh_10, mm_87, getitem_103, view_346, getitem_105, add_94, rsqrt_26, view_348, getitem_107, view_366, getitem_109, add_97, rsqrt_27, view_368, getitem_111, view_386, getitem_113, add_100, rsqrt_28, view_388, mm_97, tanh_11, mm_98, getitem_115, view_392, getitem_117, add_104, rsqrt_29, view_394, getitem_119, view_412, getitem_121, add_107, rsqrt_30, view_414, getitem_123, view_432, getitem_125, add_110, rsqrt_31, view_434, mm_108, tanh_12, mm_109, getitem_127, view_438, getitem_129, add_114, rsqrt_32, view_440, getitem_131, view_458, getitem_133, add_117, rsqrt_33, view_460, getitem_135, view_478, getitem_137, add_120, rsqrt_34, view_480, mm_119, tanh_13, mm_120, getitem_139, view_484, getitem_141, add_124, rsqrt_35, view_486, getitem_143, view_504, getitem_145, add_127, rsqrt_36, view_506, getitem_147, view_524, getitem_149, add_130, rsqrt_37, view_526, mm_130, tanh_14, mm_131, getitem_151, view_530, getitem_153, add_134, rsqrt_38, view_532, getitem_155, view_550, getitem_157, add_137, rsqrt_39, view_552, getitem_159, view_570, getitem_161, add_140, rsqrt_40, view_572, mm_141, tanh_15, mm_142, getitem_163, view_576, getitem_165, add_144, rsqrt_41, getitem_167, view_578, sub_30, convert_element_type_7, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, alias_87, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, alias_89, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, alias_93, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, alias_95, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, alias_99, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, alias_101, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, alias_105, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, alias_107, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, alias_111, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, alias_113, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, alias_117, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, alias_119, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, alias_123, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, alias_125, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, alias_129, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, alias_131, permute_750, permute_751, permute_756, permute_761, permute_766, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, alias_136, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, alias_140, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, alias_144, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, alias_148, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, alias_152, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, alias_156, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, alias_160, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, alias_164, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MT5ForConditionalGeneration', benchmark_compiled_module)
