
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


# kernel path: /tmp/torchinductor_youkaichao/vp/cvpt7aie3wltvc3zrncnd6pip4vxq435k7erf63qtiew3o6wupi4.py
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
    xnumel = 32899072
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


# kernel path: /tmp/torchinductor_youkaichao/24/c24ehugbhq4mk4kqjd5l3f7rgbwpj6e3pixv2bkokp455tpd7tk7.py
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
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrd6sunyvbpci42oewpyut7p7sbodwoukd66acbgv2r7rrgwxkm.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 32128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr4 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.full([1, 1], -100, tl.int64)
        tmp18 = tmp1 != tmp17
        tmp23 = tmp20 / tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp18, tmp23, tmp24)
        tmp26 = tmp16 * tmp25
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 * tmp13
        tmp30 = tmp26 - tmp29
        tmp31 = tmp15 + tmp30
        tl.store(out_ptr1 + (r1 + (32128*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ql/cqlzq2qw2nntmagm54og7iakjebcvdboqnqrkiaongckys6pnzdy.py
# Source Nodes: [hidden_states_186], Original ATen: [aten.mul, aten.native_dropout_backward, aten.sum]
# hidden_states_186 => mul_69
triton_per_fused_mul_native_dropout_backward_sum_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_dropout_backward_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask).to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.04419417382415922
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/cszpcj75l7khzgyodv4i5jngkuyftl6dd7c2z7lefgki4lhuviyf.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1024
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
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp1 = 0.04419417382415922
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp9 * tmp16
    tmp18 = -0.5
    tmp19 = tmp15 * tmp18
    tmp20 = tmp16 * tmp16
    tmp21 = tmp20 * tmp16
    tmp22 = tmp19 * tmp21
    tmp23 = 512.0
    tmp24 = tmp22 / tmp23
    tmp25 = 2.0
    tmp26 = tmp10 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp17 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp5
    tmp32 = tmp28 * tmp31
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp28, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eq/ceqi7fp4wzgop25getx4uwajrrtmczyeok3mwg7nsabeg74ah4x7.py
# Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
# loss => full_default_7
triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (x0), None).to(tl.int1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp0, tmp7, tmp6)
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/cto67lrrmdvrugcro3t2b6y466drimlfvbzelajc2m5eopjpptep.py
# Source Nodes: [hidden_states_178], Original ATen: [aten.mul, aten.sum]
# hidden_states_178 => mul_67
triton_per_fused_mul_sum_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvlkrmxb7vh3dt5soj4ztospcgmqe3oeyg4yjyggofzdt76k5cy.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/hi/chi4xa27tzvmnlq3gnpde32mq3ovq4kn646ftg3qfkmi2yha2bct.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcal3geqstq5wisy27ors34wz5chtmdlujbkie77kfo3mav7jux.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]

triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9', 'mutated_arg_names': ['out_ptr2']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 8192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp13, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjlygqorvgronkr2ibajv7bll43ixcakvalehmijzley6zgq2tz.py
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
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_10', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxgwsg3yphpeq7ajkw4kgs5uhphgsltcx5ghtw6bxxr6c4pqgrs.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (512*x1)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6s7bkuykjn2dhlxhhjzelk5homglzzvyg3meqcdjwzpnujepze.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (1024*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (64*y1) + (512*y0)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czoeztw5vdfyr47kvboijswtyhq5no74adsfw7egsq3xubnmnpte.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ga/cgad545g7mjjfqbesr4ylu35cmfell3dqvwd3ikwbd5drv5bgtc3.py
# Source Nodes: [hidden_states_169], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_169 => mul_63
triton_per_fused_add_mul_sum_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6f/c6f75au4yslvuhitsa77gkxeisvmlbdhimf3gztth4lthzscu4kh.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/fu/cfu6734xdekdjaj666y246ltwzgioe2jcknqgbzv5qbgkokgaved.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*i1', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*i1', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20, 21))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1024
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
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp30 = tl.load(in_ptr13 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp38 = tl.load(in_ptr15 + (x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
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
    tmp26 = tmp25.to(tl.float32)
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tmp29 = tmp24 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp39 = tmp31 * tmp38
    tmp40 = -0.5
    tmp41 = tmp37 * tmp40
    tmp42 = tmp38 * tmp38
    tmp43 = tmp42 * tmp38
    tmp44 = tmp41 * tmp43
    tmp45 = 512.0
    tmp46 = tmp44 / tmp45
    tmp47 = 2.0
    tmp48 = tmp32 * tmp47
    tmp49 = tmp46 * tmp48
    tmp50 = tmp39 + tmp49
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 * tmp27
    tmp54 = tmp50 * tmp53
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp50, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp54, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkqkjpw37bhgjjy35qs3z5eker6xooj5suusdpai2uczzzbwycr.py
# Source Nodes: [loss], Original ATen: [aten.as_strided_scatter, aten.embedding_dense_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_17', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9 + tmp0
    tmp11 = tl.full([1], False, tl.int1)
    tmp12 = 0.0
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4y/c4y4tpbsf2zffs75pc6zcuwgo2jwgpb45gey43355nzqnhzhwjfd.py
# Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_poi_fused_embedding_dense_backward_nll_loss_forward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagtdyihjs3w72cg2tvpsmfad4hjjj35iagwzsxvfwkvjkdzgone.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum]
# loss => full_default_7
triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*i64', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/nv/cnvepiypjippfhjlsgci6yp3rdbepdezzpnndzhffjsohlxb6fa4.py
# Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_poi_fused_embedding_dense_backward_nll_loss_forward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16449536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czu2ihzloj7yzvtrj6f5ji5lg3zltl52sm4zwh6gg5jozvp6lphl.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16449536
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_134, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, getitem_7, view_23, getitem_9, add_8, rsqrt_2, view_25, getitem_11, view_43, getitem_13, add_11, rsqrt_3, view_45, getitem_15, view_47, getitem_17, add_13, rsqrt_4, view_49, getitem_19, view_67, getitem_21, add_16, rsqrt_5, view_69, getitem_23, view_71, getitem_25, add_18, rsqrt_6, view_73, getitem_27, view_91, getitem_29, add_21, rsqrt_7, view_93, getitem_31, view_95, getitem_33, add_23, rsqrt_8, view_97, getitem_35, view_115, getitem_37, add_26, rsqrt_9, view_117, getitem_39, view_119, getitem_41, add_28, rsqrt_10, view_121, getitem_43, view_139, getitem_45, add_31, rsqrt_11, view_141, getitem_47, view_143, getitem_49, add_33, rsqrt_12, getitem_51, view_145, getitem_52, getitem_53, rsqrt_13, view_146, add_37, getitem_55, view_164, getitem_57, add_40, rsqrt_14, view_166, view_169, getitem_59, view_184, getitem_61, add_44, rsqrt_15, view_186, getitem_63, view_188, getitem_65, add_46, rsqrt_16, view_190, getitem_67, view_208, getitem_69, add_49, rsqrt_17, view_210, getitem_71, view_228, getitem_73, add_52, rsqrt_18, view_230, getitem_75, view_232, getitem_77, add_54, rsqrt_19, view_234, getitem_79, view_252, getitem_81, add_57, rsqrt_20, view_254, getitem_83, view_272, getitem_85, add_60, rsqrt_21, view_274, getitem_87, view_276, getitem_89, add_62, rsqrt_22, view_278, getitem_91, view_296, getitem_93, add_65, rsqrt_23, view_298, getitem_95, view_316, getitem_97, add_68, rsqrt_24, view_318, getitem_99, view_320, getitem_101, add_70, rsqrt_25, view_322, getitem_103, view_340, getitem_105, add_73, rsqrt_26, view_342, getitem_107, view_360, getitem_109, add_76, rsqrt_27, view_362, getitem_111, view_364, getitem_113, add_78, rsqrt_28, view_366, getitem_115, view_384, getitem_117, add_81, rsqrt_29, view_386, getitem_119, view_404, getitem_121, add_84, rsqrt_30, view_406, getitem_123, view_408, getitem_125, add_86, rsqrt_31, getitem_127, view_410, sub_24, convert_element_type_7, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_67, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_69, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_73, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_75, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_79, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_81, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_85, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_87, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_91, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_93, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_97, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_99, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_104, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_108, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_112, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_116, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_120, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_124, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27 = args
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
    assert_size_stride(primals_134, (1, 1024), (1024, 1))
    assert_size_stride(view, (1, 1024), (1024, 1))
    assert_size_stride(getitem, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(getitem_1, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_1, (1024, 512), (512, 1))
    assert_size_stride(add_3, (1024, 1024), (1024, 1))
    assert_size_stride(getitem_3, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_19, (1024, 512), (512, 1))
    assert_size_stride(getitem_5, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_6, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_1, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_21, (1024, 512), (512, 1))
    assert_size_stride(getitem_7, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_23, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_9, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_8, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_2, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_25, (1024, 512), (512, 1))
    assert_size_stride(getitem_11, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_43, (1024, 512), (512, 1))
    assert_size_stride(getitem_13, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_11, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_3, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_45, (1024, 512), (512, 1))
    assert_size_stride(getitem_15, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_47, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_17, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_13, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_4, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_49, (1024, 512), (512, 1))
    assert_size_stride(getitem_19, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_67, (1024, 512), (512, 1))
    assert_size_stride(getitem_21, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_16, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_5, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_69, (1024, 512), (512, 1))
    assert_size_stride(getitem_23, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_71, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_25, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_18, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_6, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_73, (1024, 512), (512, 1))
    assert_size_stride(getitem_27, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_91, (1024, 512), (512, 1))
    assert_size_stride(getitem_29, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_21, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_7, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_93, (1024, 512), (512, 1))
    assert_size_stride(getitem_31, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_95, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_33, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_23, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_8, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_97, (1024, 512), (512, 1))
    assert_size_stride(getitem_35, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_115, (1024, 512), (512, 1))
    assert_size_stride(getitem_37, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_26, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_9, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_117, (1024, 512), (512, 1))
    assert_size_stride(getitem_39, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_119, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_41, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_28, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_10, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_121, (1024, 512), (512, 1))
    assert_size_stride(getitem_43, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_139, (1024, 512), (512, 1))
    assert_size_stride(getitem_45, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_31, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_11, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_141, (1024, 512), (512, 1))
    assert_size_stride(getitem_47, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_143, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_49, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_33, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_12, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(getitem_51, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(view_145, (1, 1024), (1024, 1))
    assert_size_stride(getitem_52, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(getitem_53, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_13, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_146, (1024, 512), (512, 1))
    assert_size_stride(add_37, (1024, 1024), (1024, 1))
    assert_size_stride(getitem_55, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_164, (1024, 512), (512, 1))
    assert_size_stride(getitem_57, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_40, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_14, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_166, (1024, 512), (512, 1))
    assert_size_stride(view_169, (1024, 512), (512, 1))
    assert_size_stride(getitem_59, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_184, (1024, 512), (512, 1))
    assert_size_stride(getitem_61, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_44, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_15, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_186, (1024, 512), (512, 1))
    assert_size_stride(getitem_63, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_188, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_65, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_46, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_16, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_190, (1024, 512), (512, 1))
    assert_size_stride(getitem_67, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_208, (1024, 512), (512, 1))
    assert_size_stride(getitem_69, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_49, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_17, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_210, (1024, 512), (512, 1))
    assert_size_stride(getitem_71, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_228, (1024, 512), (512, 1))
    assert_size_stride(getitem_73, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_52, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_18, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_230, (1024, 512), (512, 1))
    assert_size_stride(getitem_75, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_232, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_77, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_54, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_19, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_234, (1024, 512), (512, 1))
    assert_size_stride(getitem_79, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_252, (1024, 512), (512, 1))
    assert_size_stride(getitem_81, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_57, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_20, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_254, (1024, 512), (512, 1))
    assert_size_stride(getitem_83, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_272, (1024, 512), (512, 1))
    assert_size_stride(getitem_85, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_60, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_21, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_274, (1024, 512), (512, 1))
    assert_size_stride(getitem_87, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_276, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_89, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_62, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_22, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_278, (1024, 512), (512, 1))
    assert_size_stride(getitem_91, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_296, (1024, 512), (512, 1))
    assert_size_stride(getitem_93, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_65, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_23, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_298, (1024, 512), (512, 1))
    assert_size_stride(getitem_95, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_316, (1024, 512), (512, 1))
    assert_size_stride(getitem_97, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_68, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_24, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_318, (1024, 512), (512, 1))
    assert_size_stride(getitem_99, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_320, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_101, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_70, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_25, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_322, (1024, 512), (512, 1))
    assert_size_stride(getitem_103, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_340, (1024, 512), (512, 1))
    assert_size_stride(getitem_105, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_73, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_26, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_342, (1024, 512), (512, 1))
    assert_size_stride(getitem_107, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_360, (1024, 512), (512, 1))
    assert_size_stride(getitem_109, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_76, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_27, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_362, (1024, 512), (512, 1))
    assert_size_stride(getitem_111, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_364, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_113, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_78, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_28, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_366, (1024, 512), (512, 1))
    assert_size_stride(getitem_115, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_384, (1024, 512), (512, 1))
    assert_size_stride(getitem_117, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_81, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_29, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_386, (1024, 512), (512, 1))
    assert_size_stride(getitem_119, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_404, (1024, 512), (512, 1))
    assert_size_stride(getitem_121, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_84, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_30, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_406, (1024, 512), (512, 1))
    assert_size_stride(getitem_123, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_408, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_125, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_86, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_31, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(getitem_127, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(view_410, (1024, 512), (512, 1))
    assert_size_stride(sub_24, (1024, 32128), (32128, 1))
    assert_size_stride(convert_element_type_7, (), ())
    assert_size_stride(permute_191, (32128, 512), (512, 1))
    assert_size_stride(permute_195, (512, 2048), (2048, 1))
    assert_size_stride(le_1, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_199, (2048, 512), (512, 1))
    assert_size_stride(permute_203, (512, 512), (512, 1))
    assert_size_stride(permute_206, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_207, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_67, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_208, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_209, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_214, (512, 512), (512, 1))
    assert_size_stride(permute_219, (512, 512), (512, 1))
    assert_size_stride(permute_224, (512, 512), (512, 1))
    assert_size_stride(permute_228, (512, 512), (512, 1))
    assert_size_stride(permute_231, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_232, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_69, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_233, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_234, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_239, (512, 512), (512, 1))
    assert_size_stride(permute_244, (512, 512), (512, 1))
    assert_size_stride(permute_249, (512, 512), (512, 1))
    assert_size_stride(permute_253, (512, 2048), (2048, 1))
    assert_size_stride(le_2, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_257, (2048, 512), (512, 1))
    assert_size_stride(permute_261, (512, 512), (512, 1))
    assert_size_stride(permute_264, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_265, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_73, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_266, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_267, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_272, (512, 512), (512, 1))
    assert_size_stride(permute_277, (512, 512), (512, 1))
    assert_size_stride(permute_282, (512, 512), (512, 1))
    assert_size_stride(permute_286, (512, 512), (512, 1))
    assert_size_stride(permute_289, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_290, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_75, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_291, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_292, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_297, (512, 512), (512, 1))
    assert_size_stride(permute_302, (512, 512), (512, 1))
    assert_size_stride(permute_307, (512, 512), (512, 1))
    assert_size_stride(permute_311, (512, 2048), (2048, 1))
    assert_size_stride(le_3, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_315, (2048, 512), (512, 1))
    assert_size_stride(permute_319, (512, 512), (512, 1))
    assert_size_stride(permute_322, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_323, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_79, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_324, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_325, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_330, (512, 512), (512, 1))
    assert_size_stride(permute_335, (512, 512), (512, 1))
    assert_size_stride(permute_340, (512, 512), (512, 1))
    assert_size_stride(permute_344, (512, 512), (512, 1))
    assert_size_stride(permute_347, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_348, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_81, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_349, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_350, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_355, (512, 512), (512, 1))
    assert_size_stride(permute_360, (512, 512), (512, 1))
    assert_size_stride(permute_365, (512, 512), (512, 1))
    assert_size_stride(permute_369, (512, 2048), (2048, 1))
    assert_size_stride(le_4, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_373, (2048, 512), (512, 1))
    assert_size_stride(permute_377, (512, 512), (512, 1))
    assert_size_stride(permute_380, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_381, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_85, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_382, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_383, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_388, (512, 512), (512, 1))
    assert_size_stride(permute_393, (512, 512), (512, 1))
    assert_size_stride(permute_398, (512, 512), (512, 1))
    assert_size_stride(permute_402, (512, 512), (512, 1))
    assert_size_stride(permute_405, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_406, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_87, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_407, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_408, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_413, (512, 512), (512, 1))
    assert_size_stride(permute_418, (512, 512), (512, 1))
    assert_size_stride(permute_423, (512, 512), (512, 1))
    assert_size_stride(permute_427, (512, 2048), (2048, 1))
    assert_size_stride(le_5, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_431, (2048, 512), (512, 1))
    assert_size_stride(permute_435, (512, 512), (512, 1))
    assert_size_stride(permute_438, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_439, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_91, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_440, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_441, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_446, (512, 512), (512, 1))
    assert_size_stride(permute_451, (512, 512), (512, 1))
    assert_size_stride(permute_456, (512, 512), (512, 1))
    assert_size_stride(permute_460, (512, 512), (512, 1))
    assert_size_stride(permute_463, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_464, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_93, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_465, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_466, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_471, (512, 512), (512, 1))
    assert_size_stride(permute_476, (512, 512), (512, 1))
    assert_size_stride(permute_481, (512, 512), (512, 1))
    assert_size_stride(permute_485, (512, 2048), (2048, 1))
    assert_size_stride(le_6, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_489, (2048, 512), (512, 1))
    assert_size_stride(permute_493, (512, 512), (512, 1))
    assert_size_stride(permute_496, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_497, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_97, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_498, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_499, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_504, (512, 512), (512, 1))
    assert_size_stride(permute_509, (512, 512), (512, 1))
    assert_size_stride(permute_514, (512, 512), (512, 1))
    assert_size_stride(permute_518, (512, 512), (512, 1))
    assert_size_stride(permute_521, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_522, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_99, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_524, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_525, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_530, (512, 512), (512, 1))
    assert_size_stride(permute_535, (512, 512), (512, 1))
    assert_size_stride(permute_540, (512, 512), (512, 1))
    assert_size_stride(permute_544, (512, 2048), (2048, 1))
    assert_size_stride(le_7, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_548, (2048, 512), (512, 1))
    assert_size_stride(permute_552, (512, 512), (512, 1))
    assert_size_stride(permute_555, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_556, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_104, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_557, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_558, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_563, (512, 512), (512, 1))
    assert_size_stride(permute_568, (512, 512), (512, 1))
    assert_size_stride(permute_573, (512, 512), (512, 1))
    assert_size_stride(permute_577, (512, 2048), (2048, 1))
    assert_size_stride(le_8, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_581, (2048, 512), (512, 1))
    assert_size_stride(permute_585, (512, 512), (512, 1))
    assert_size_stride(permute_588, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_589, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_108, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_590, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_591, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_596, (512, 512), (512, 1))
    assert_size_stride(permute_601, (512, 512), (512, 1))
    assert_size_stride(permute_606, (512, 512), (512, 1))
    assert_size_stride(permute_610, (512, 2048), (2048, 1))
    assert_size_stride(le_9, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_614, (2048, 512), (512, 1))
    assert_size_stride(permute_618, (512, 512), (512, 1))
    assert_size_stride(permute_621, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_622, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_112, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_623, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_624, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_629, (512, 512), (512, 1))
    assert_size_stride(permute_634, (512, 512), (512, 1))
    assert_size_stride(permute_639, (512, 512), (512, 1))
    assert_size_stride(permute_643, (512, 2048), (2048, 1))
    assert_size_stride(le_10, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_647, (2048, 512), (512, 1))
    assert_size_stride(permute_651, (512, 512), (512, 1))
    assert_size_stride(permute_654, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_655, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_116, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_656, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_657, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_662, (512, 512), (512, 1))
    assert_size_stride(permute_667, (512, 512), (512, 1))
    assert_size_stride(permute_672, (512, 512), (512, 1))
    assert_size_stride(permute_676, (512, 2048), (2048, 1))
    assert_size_stride(le_11, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_680, (2048, 512), (512, 1))
    assert_size_stride(permute_684, (512, 512), (512, 1))
    assert_size_stride(permute_687, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_688, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_120, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_689, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_690, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_695, (512, 512), (512, 1))
    assert_size_stride(permute_700, (512, 512), (512, 1))
    assert_size_stride(permute_705, (512, 512), (512, 1))
    assert_size_stride(permute_709, (512, 2048), (2048, 1))
    assert_size_stride(le_12, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_713, (2048, 512), (512, 1))
    assert_size_stride(permute_717, (512, 512), (512, 1))
    assert_size_stride(permute_720, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_721, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_124, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_723, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_724, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_729, (512, 512), (512, 1))
    assert_size_stride(permute_734, (512, 512), (512, 1))
    assert_size_stride(permute_739, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 1024, 32128), (32899072, 32128, 1))
    assert_size_stride(tangents_3, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_4, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_5, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_6, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_7, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_8, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_9, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_10, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_11, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_12, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_13, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_14, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_15, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_16, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_17, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_18, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_19, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_20, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_21, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_22, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_23, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_24, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_25, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_26, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_27, (1, 1024, 512), (524288, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1024, 32128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 32899072, grid=grid(32899072), stream=stream0)
        buf1 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_134, buf1, 1024, grid=grid(1024), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 1024, 32128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_134, tangents_1, convert_element_type_7, tangents_2, sub_24, buf5, 1024, 32128, grid=grid(1024), stream=stream0)
        del buf0
        del convert_element_type_7
        del primals_134
        del sub_24
        del tangents_1
        del tangents_2
        buf6 = empty((32128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (32128, 1024), (1, 32128), 0), view_410, out=buf6)
        del view_410
        buf7 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1024, 32128), (32128, 1), 0), permute_191, out=buf7)
        del buf5
        del permute_191
        buf8 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_186], Original ATen: [aten.mul, aten.native_dropout_backward, aten.sum]
        triton_per_fused_mul_native_dropout_backward_sum_3.run(buf7, getitem_127, add_86, rsqrt_31, buf8, 512, 1024, grid=grid(512), stream=stream0)
        buf10 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        buf11 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_4.run(buf7, getitem_127, primals_32, add_86, rsqrt_31, getitem_125, buf10, buf11, 1024, 512, grid=grid(1024), stream=stream0)
        del add_86
        del getitem_125
        del getitem_127
        del primals_32
        del rsqrt_31
        buf12 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (512, 1024), (1, 512), 0), view_408, out=buf12)
        del view_408
        buf13 = empty((1024, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (1024, 512), (512, 1), 0), permute_195, out=buf13)
        del permute_195
        buf14 = reinterpret_tensor(buf13, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf13  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf14, le_1, getitem_123, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_123
        del le_1
        buf15 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (2048, 1024), (1, 2048), 0), view_406, out=buf15)
        del view_406
        buf16 = reinterpret_tensor(buf11, (1024, 512), (512, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (1024, 2048), (2048, 1), 0), permute_199, out=buf16)
        del permute_199
        buf17 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_178], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf16, add_84, rsqrt_30, buf17, 512, 1024, grid=grid(512), stream=stream0)
        buf19 = buf10; del buf10  # reuse
        buf20 = reinterpret_tensor(buf7, (1, 1024, 512), (524288, 512, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf19, buf16, primals_31, add_84, rsqrt_30, getitem_121, buf20, 1024, 512, grid=grid(1024), stream=stream0)
        del add_84
        del getitem_121
        del primals_31
        del rsqrt_30
        buf21 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 1024), (1, 512), 0), view_404, out=buf21)
        del view_404
        buf22 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 512), (512, 1), 0), permute_203, out=buf22)
        del permute_203
        buf23 = reinterpret_tensor(buf20, (8, 1024, 64), (65536, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_206, reinterpret_tensor(buf22, (8, 1024, 64), (64, 512, 1), 0), out=buf23)
        del permute_206
        buf24 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf22, (8, 1024, 64), (64, 512, 1), 0), permute_207, out=buf24)
        del permute_207
        buf28 = empty((8388608, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf28, 8388608, grid=grid(8388608), stream=stream0)
        buf31 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf24, getitem_119, alias_67, buf28, buf31, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_67
        del getitem_119
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf28, buf31, 8388608, grid=grid(8388608), stream=stream0)
        buf33 = reinterpret_tensor(buf22, (8, 64, 1024), (65536, 1024, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_208, buf31, out=buf33)
        del permute_208
        buf34 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf31, permute_209, out=buf34)
        del permute_209
        buf35 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_26, buf23, buf35, 524288, grid=grid(524288), stream=stream0)
        del tangents_26
        buf36 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (512, 1024), (1, 512), 0), view_169, out=buf36)
        buf37 = reinterpret_tensor(buf23, (1024, 512), (512, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (1024, 512), (512, 1), 0), permute_214, out=buf37)
        del permute_214
        buf38 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_25, buf33, buf38, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_25
        buf39 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (512, 1024), (1, 512), 0), view_169, out=buf39)
        buf40 = reinterpret_tensor(buf33, (1024, 512), (512, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (1024, 512), (512, 1), 0), permute_219, out=buf40)
        del permute_219
        buf41 = reinterpret_tensor(buf38, (1024, 512), (512, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf34, buf41, 524288, grid=grid(524288), stream=stream0)
        buf42 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (512, 1024), (1, 512), 0), view_386, out=buf42)
        del view_386
        buf43 = reinterpret_tensor(buf34, (1024, 512), (512, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf41, permute_224, out=buf43)
        del permute_224
        buf44 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_174], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf43, add_81, rsqrt_29, buf44, 512, 1024, grid=grid(512), stream=stream0)
        buf46 = buf19; del buf19  # reuse
        buf47 = reinterpret_tensor(buf41, (1, 1024, 512), (524288, 512, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf46, buf43, primals_30, add_81, rsqrt_29, getitem_117, buf47, 1024, 512, grid=grid(1024), stream=stream0)
        del add_81
        del getitem_117
        del primals_30
        del rsqrt_29
        buf48 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (512, 1024), (1, 512), 0), view_384, out=buf48)
        del view_384
        buf49 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1024, 512), (512, 1), 0), permute_228, out=buf49)
        del permute_228
        buf50 = reinterpret_tensor(buf47, (8, 1024, 64), (65536, 64, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_231, reinterpret_tensor(buf49, (8, 1024, 64), (64, 512, 1), 0), out=buf50)
        del permute_231
        buf51 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (8, 1024, 64), (64, 512, 1), 0), permute_232, out=buf51)
        del permute_232
        buf54 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf54, 8388608, grid=grid(8388608), stream=stream0)
        buf57 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf51, getitem_115, alias_69, buf54, buf57, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_69
        del getitem_115
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf54, buf57, 8388608, grid=grid(8388608), stream=stream0)
        buf59 = reinterpret_tensor(buf49, (8, 64, 1024), (65536, 1024, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_233, buf57, out=buf59)
        del permute_233
        buf60 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf57, permute_234, out=buf60)
        del permute_234
        buf61 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_24, buf50, buf61, 524288, grid=grid(524288), stream=stream0)
        del tangents_24
        buf62 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (512, 1024), (1, 512), 0), view_366, out=buf62)
        buf63 = reinterpret_tensor(buf50, (1024, 512), (512, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1024, 512), (512, 1), 0), permute_239, out=buf63)
        del permute_239
        buf64 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_23, buf59, buf64, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_23
        buf65 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 1024), (1, 512), 0), view_366, out=buf65)
        buf66 = reinterpret_tensor(buf59, (1024, 512), (512, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (1024, 512), (512, 1), 0), permute_244, out=buf66)
        del permute_244
        buf67 = reinterpret_tensor(buf64, (1024, 512), (512, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf60, buf67, 524288, grid=grid(524288), stream=stream0)
        buf68 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (512, 1024), (1, 512), 0), view_366, out=buf68)
        del view_366
        buf69 = reinterpret_tensor(buf60, (1024, 512), (512, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf67, permute_249, out=buf69)
        del permute_249
        buf70 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_169], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf63, buf66, buf69, add_78, rsqrt_28, buf70, 512, 1024, grid=grid(512), stream=stream0)
        buf72 = buf46; del buf46  # reuse
        buf73 = reinterpret_tensor(buf67, (1, 1024, 512), (524288, 512, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf72, buf63, buf66, buf69, primals_29, add_78, rsqrt_28, getitem_113, buf73, 1024, 512, grid=grid(1024), stream=stream0)
        del add_78
        del getitem_113
        del primals_29
        del rsqrt_28
        buf74 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (512, 1024), (1, 512), 0), view_364, out=buf74)
        del view_364
        buf75 = reinterpret_tensor(buf14, (1024, 2048), (2048, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (1024, 512), (512, 1), 0), permute_253, out=buf75)
        del permute_253
        buf76 = reinterpret_tensor(buf75, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf75  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf76, le_2, getitem_111, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_111
        del le_2
        buf77 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (2048, 1024), (1, 2048), 0), view_362, out=buf77)
        del view_362
        buf78 = reinterpret_tensor(buf73, (1024, 512), (512, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (1024, 2048), (2048, 1), 0), permute_257, out=buf78)
        del permute_257
        buf79 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_161], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf78, add_76, rsqrt_27, buf79, 512, 1024, grid=grid(512), stream=stream0)
        buf81 = buf72; del buf72  # reuse
        buf82 = reinterpret_tensor(buf69, (1, 1024, 512), (524288, 512, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf81, buf78, primals_28, add_76, rsqrt_27, getitem_109, buf82, 1024, 512, grid=grid(1024), stream=stream0)
        del add_76
        del getitem_109
        del primals_28
        del rsqrt_27
        buf83 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (512, 1024), (1, 512), 0), view_360, out=buf83)
        del view_360
        buf84 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 512), (512, 1), 0), permute_261, out=buf84)
        del permute_261
        buf85 = reinterpret_tensor(buf82, (8, 1024, 64), (65536, 64, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_264, reinterpret_tensor(buf84, (8, 1024, 64), (64, 512, 1), 0), out=buf85)
        del permute_264
        buf86 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (8, 1024, 64), (64, 512, 1), 0), permute_265, out=buf86)
        del permute_265
        buf89 = reinterpret_tensor(buf51, (8388608, ), (1, ), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf89, 8388608, grid=grid(8388608), stream=stream0)
        buf92 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf86, getitem_107, alias_73, buf89, buf92, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_73
        del getitem_107
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf89, buf92, 8388608, grid=grid(8388608), stream=stream0)
        buf94 = reinterpret_tensor(buf84, (8, 64, 1024), (65536, 1024, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_266, buf92, out=buf94)
        del permute_266
        buf95 = reinterpret_tensor(buf66, (8, 1024, 64), (65536, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf92, permute_267, out=buf95)
        del permute_267
        buf96 = reinterpret_tensor(buf63, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_22, buf85, buf96, 524288, grid=grid(524288), stream=stream0)
        del tangents_22
        buf97 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (512, 1024), (1, 512), 0), view_169, out=buf97)
        buf98 = reinterpret_tensor(buf85, (1024, 512), (512, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (1024, 512), (512, 1), 0), permute_272, out=buf98)
        del permute_272
        buf99 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_21, buf94, buf99, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_21
        buf100 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 1024), (1, 512), 0), view_169, out=buf100)
        buf101 = reinterpret_tensor(buf94, (1024, 512), (512, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 512), (512, 1), 0), permute_277, out=buf101)
        del permute_277
        buf102 = reinterpret_tensor(buf99, (1024, 512), (512, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf95, buf102, 524288, grid=grid(524288), stream=stream0)
        buf103 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (512, 1024), (1, 512), 0), view_342, out=buf103)
        del view_342
        buf104 = reinterpret_tensor(buf95, (1024, 512), (512, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf102, permute_282, out=buf104)
        del permute_282
        buf105 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_157], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf104, add_73, rsqrt_26, buf105, 512, 1024, grid=grid(512), stream=stream0)
        buf107 = buf81; del buf81  # reuse
        buf108 = reinterpret_tensor(buf102, (1, 1024, 512), (524288, 512, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf107, buf104, primals_27, add_73, rsqrt_26, getitem_105, buf108, 1024, 512, grid=grid(1024), stream=stream0)
        del add_73
        del getitem_105
        del primals_27
        del rsqrt_26
        buf109 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (512, 1024), (1, 512), 0), view_340, out=buf109)
        del view_340
        buf110 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (1024, 512), (512, 1), 0), permute_286, out=buf110)
        del permute_286
        buf111 = reinterpret_tensor(buf108, (8, 1024, 64), (65536, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_289, reinterpret_tensor(buf110, (8, 1024, 64), (64, 512, 1), 0), out=buf111)
        del permute_289
        buf112 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (8, 1024, 64), (64, 512, 1), 0), permute_290, out=buf112)
        del permute_290
        buf115 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf115, 8388608, grid=grid(8388608), stream=stream0)
        buf118 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf112, getitem_103, alias_75, buf115, buf118, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_75
        del getitem_103
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf115, buf118, 8388608, grid=grid(8388608), stream=stream0)
        buf120 = reinterpret_tensor(buf110, (8, 64, 1024), (65536, 1024, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_291, buf118, out=buf120)
        del permute_291
        buf121 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf118, permute_292, out=buf121)
        del permute_292
        buf122 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_20, buf111, buf122, 524288, grid=grid(524288), stream=stream0)
        del tangents_20
        buf123 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (512, 1024), (1, 512), 0), view_322, out=buf123)
        buf124 = reinterpret_tensor(buf111, (1024, 512), (512, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), permute_297, out=buf124)
        del permute_297
        buf125 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_19, buf120, buf125, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_19
        buf126 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 1024), (1, 512), 0), view_322, out=buf126)
        buf127 = reinterpret_tensor(buf120, (1024, 512), (512, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1024, 512), (512, 1), 0), permute_302, out=buf127)
        del permute_302
        buf128 = reinterpret_tensor(buf125, (1024, 512), (512, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf121, buf128, 524288, grid=grid(524288), stream=stream0)
        buf129 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 1024), (1, 512), 0), view_322, out=buf129)
        del view_322
        buf130 = reinterpret_tensor(buf121, (1024, 512), (512, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf128, permute_307, out=buf130)
        del permute_307
        buf131 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_152], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf124, buf127, buf130, add_70, rsqrt_25, buf131, 512, 1024, grid=grid(512), stream=stream0)
        buf133 = buf107; del buf107  # reuse
        buf134 = reinterpret_tensor(buf128, (1, 1024, 512), (524288, 512, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf133, buf124, buf127, buf130, primals_26, add_70, rsqrt_25, getitem_101, buf134, 1024, 512, grid=grid(1024), stream=stream0)
        del add_70
        del getitem_101
        del primals_26
        del rsqrt_25
        buf135 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (512, 1024), (1, 512), 0), view_320, out=buf135)
        del view_320
        buf136 = reinterpret_tensor(buf76, (1024, 2048), (2048, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (1024, 512), (512, 1), 0), permute_311, out=buf136)
        del permute_311
        buf137 = reinterpret_tensor(buf136, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf136  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf137, le_3, getitem_99, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_99
        del le_3
        buf138 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (2048, 1024), (1, 2048), 0), view_318, out=buf138)
        del view_318
        buf139 = reinterpret_tensor(buf134, (1024, 512), (512, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (1024, 2048), (2048, 1), 0), permute_315, out=buf139)
        del permute_315
        buf140 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_144], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf139, add_68, rsqrt_24, buf140, 512, 1024, grid=grid(512), stream=stream0)
        buf142 = buf133; del buf133  # reuse
        buf143 = reinterpret_tensor(buf130, (1, 1024, 512), (524288, 512, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf142, buf139, primals_25, add_68, rsqrt_24, getitem_97, buf143, 1024, 512, grid=grid(1024), stream=stream0)
        del add_68
        del getitem_97
        del primals_25
        del rsqrt_24
        buf144 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 1024), (1, 512), 0), view_316, out=buf144)
        del view_316
        buf145 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (1024, 512), (512, 1), 0), permute_319, out=buf145)
        del permute_319
        buf146 = reinterpret_tensor(buf143, (8, 1024, 64), (65536, 64, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_322, reinterpret_tensor(buf145, (8, 1024, 64), (64, 512, 1), 0), out=buf146)
        del permute_322
        buf147 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (8, 1024, 64), (64, 512, 1), 0), permute_323, out=buf147)
        del permute_323
        buf150 = reinterpret_tensor(buf112, (8388608, ), (1, ), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf150, 8388608, grid=grid(8388608), stream=stream0)
        buf153 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf147, getitem_95, alias_79, buf150, buf153, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_79
        del getitem_95
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf150, buf153, 8388608, grid=grid(8388608), stream=stream0)
        buf155 = reinterpret_tensor(buf145, (8, 64, 1024), (65536, 1024, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_324, buf153, out=buf155)
        del permute_324
        buf156 = reinterpret_tensor(buf127, (8, 1024, 64), (65536, 64, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf153, permute_325, out=buf156)
        del permute_325
        buf157 = reinterpret_tensor(buf124, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_18, buf146, buf157, 524288, grid=grid(524288), stream=stream0)
        del tangents_18
        buf158 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (512, 1024), (1, 512), 0), view_169, out=buf158)
        buf159 = reinterpret_tensor(buf146, (1024, 512), (512, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (1024, 512), (512, 1), 0), permute_330, out=buf159)
        del permute_330
        buf160 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_17, buf155, buf160, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_17
        buf161 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 1024), (1, 512), 0), view_169, out=buf161)
        buf162 = reinterpret_tensor(buf155, (1024, 512), (512, 1), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (1024, 512), (512, 1), 0), permute_335, out=buf162)
        del permute_335
        buf163 = reinterpret_tensor(buf160, (1024, 512), (512, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf156, buf163, 524288, grid=grid(524288), stream=stream0)
        buf164 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (512, 1024), (1, 512), 0), view_298, out=buf164)
        del view_298
        buf165 = reinterpret_tensor(buf156, (1024, 512), (512, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf163, permute_340, out=buf165)
        del permute_340
        buf166 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_140], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf165, add_65, rsqrt_23, buf166, 512, 1024, grid=grid(512), stream=stream0)
        buf168 = buf142; del buf142  # reuse
        buf169 = reinterpret_tensor(buf163, (1, 1024, 512), (524288, 512, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf168, buf165, primals_24, add_65, rsqrt_23, getitem_93, buf169, 1024, 512, grid=grid(1024), stream=stream0)
        del add_65
        del getitem_93
        del primals_24
        del rsqrt_23
        buf170 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (512, 1024), (1, 512), 0), view_296, out=buf170)
        del view_296
        buf171 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 512), (512, 1), 0), permute_344, out=buf171)
        del permute_344
        buf172 = reinterpret_tensor(buf169, (8, 1024, 64), (65536, 64, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_347, reinterpret_tensor(buf171, (8, 1024, 64), (64, 512, 1), 0), out=buf172)
        del permute_347
        buf173 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (8, 1024, 64), (64, 512, 1), 0), permute_348, out=buf173)
        del permute_348
        buf176 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf176, 8388608, grid=grid(8388608), stream=stream0)
        buf179 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf173, getitem_91, alias_81, buf176, buf179, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_81
        del getitem_91
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf176, buf179, 8388608, grid=grid(8388608), stream=stream0)
        buf181 = reinterpret_tensor(buf171, (8, 64, 1024), (65536, 1024, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_349, buf179, out=buf181)
        del permute_349
        buf182 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf179, permute_350, out=buf182)
        del permute_350
        buf183 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_16, buf172, buf183, 524288, grid=grid(524288), stream=stream0)
        del tangents_16
        buf184 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (512, 1024), (1, 512), 0), view_278, out=buf184)
        buf185 = reinterpret_tensor(buf172, (1024, 512), (512, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (1024, 512), (512, 1), 0), permute_355, out=buf185)
        del permute_355
        buf186 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_15, buf181, buf186, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_15
        buf187 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (512, 1024), (1, 512), 0), view_278, out=buf187)
        buf188 = reinterpret_tensor(buf181, (1024, 512), (512, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (1024, 512), (512, 1), 0), permute_360, out=buf188)
        del permute_360
        buf189 = reinterpret_tensor(buf186, (1024, 512), (512, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf182, buf189, 524288, grid=grid(524288), stream=stream0)
        buf190 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (512, 1024), (1, 512), 0), view_278, out=buf190)
        del view_278
        buf191 = reinterpret_tensor(buf182, (1024, 512), (512, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf189, permute_365, out=buf191)
        del permute_365
        buf192 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_135], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf185, buf188, buf191, add_62, rsqrt_22, buf192, 512, 1024, grid=grid(512), stream=stream0)
        buf194 = buf168; del buf168  # reuse
        buf195 = reinterpret_tensor(buf189, (1, 1024, 512), (524288, 512, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf194, buf185, buf188, buf191, primals_23, add_62, rsqrt_22, getitem_89, buf195, 1024, 512, grid=grid(1024), stream=stream0)
        del add_62
        del getitem_89
        del primals_23
        del rsqrt_22
        buf196 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 1024), (1, 512), 0), view_276, out=buf196)
        del view_276
        buf197 = reinterpret_tensor(buf137, (1024, 2048), (2048, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 512), (512, 1), 0), permute_369, out=buf197)
        del permute_369
        buf198 = reinterpret_tensor(buf197, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf197  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf198, le_4, getitem_87, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_87
        del le_4
        buf199 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (2048, 1024), (1, 2048), 0), view_274, out=buf199)
        del view_274
        buf200 = reinterpret_tensor(buf195, (1024, 512), (512, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (1024, 2048), (2048, 1), 0), permute_373, out=buf200)
        del permute_373
        buf201 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_127], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf200, add_60, rsqrt_21, buf201, 512, 1024, grid=grid(512), stream=stream0)
        buf203 = buf194; del buf194  # reuse
        buf204 = reinterpret_tensor(buf191, (1, 1024, 512), (524288, 512, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf203, buf200, primals_22, add_60, rsqrt_21, getitem_85, buf204, 1024, 512, grid=grid(1024), stream=stream0)
        del add_60
        del getitem_85
        del primals_22
        del rsqrt_21
        buf205 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (512, 1024), (1, 512), 0), view_272, out=buf205)
        del view_272
        buf206 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (1024, 512), (512, 1), 0), permute_377, out=buf206)
        del permute_377
        buf207 = reinterpret_tensor(buf204, (8, 1024, 64), (65536, 64, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_380, reinterpret_tensor(buf206, (8, 1024, 64), (64, 512, 1), 0), out=buf207)
        del permute_380
        buf208 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf206, (8, 1024, 64), (64, 512, 1), 0), permute_381, out=buf208)
        del permute_381
        buf211 = reinterpret_tensor(buf173, (8388608, ), (1, ), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf211, 8388608, grid=grid(8388608), stream=stream0)
        buf214 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf208, getitem_83, alias_85, buf211, buf214, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_85
        del getitem_83
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf211, buf214, 8388608, grid=grid(8388608), stream=stream0)
        buf216 = reinterpret_tensor(buf206, (8, 64, 1024), (65536, 1024, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_382, buf214, out=buf216)
        del permute_382
        buf217 = reinterpret_tensor(buf188, (8, 1024, 64), (65536, 64, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf214, permute_383, out=buf217)
        del permute_383
        buf218 = reinterpret_tensor(buf185, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_14, buf207, buf218, 524288, grid=grid(524288), stream=stream0)
        del tangents_14
        buf219 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf218, (512, 1024), (1, 512), 0), view_169, out=buf219)
        buf220 = reinterpret_tensor(buf207, (1024, 512), (512, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf218, (1024, 512), (512, 1), 0), permute_388, out=buf220)
        del permute_388
        buf221 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_13, buf216, buf221, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_13
        buf222 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (512, 1024), (1, 512), 0), view_169, out=buf222)
        buf223 = reinterpret_tensor(buf216, (1024, 512), (512, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (1024, 512), (512, 1), 0), permute_393, out=buf223)
        del permute_393
        buf225 = reinterpret_tensor(buf221, (1024, 512), (512, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf217, buf225, 524288, grid=grid(524288), stream=stream0)
        buf227 = reinterpret_tensor(buf217, (1024, 512), (512, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf225, permute_398, out=buf227)
        del permute_398
        buf230 = buf203; del buf203  # reuse
        buf231 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf230, buf227, primals_21, add_57, rsqrt_20, getitem_81, buf231, 1024, 512, grid=grid(1024), stream=stream0)
        del getitem_81
        del primals_21
        buf233 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 512), (512, 1), 0), permute_402, out=buf233)
        del permute_402
        buf234 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_405, reinterpret_tensor(buf233, (8, 1024, 64), (64, 512, 1), 0), out=buf234)
        del permute_405
        buf245 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_12, buf234, buf245, 524288, grid=grid(524288), stream=stream0)
        del tangents_12
        buf247 = reinterpret_tensor(buf234, (1024, 512), (512, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (1024, 512), (512, 1), 0), permute_413, out=buf247)
        del permute_413
        buf235 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (8, 1024, 64), (64, 512, 1), 0), permute_406, out=buf235)
        del permute_406
        buf238 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf238, 8388608, grid=grid(8388608), stream=stream0)
        buf241 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf235, getitem_79, alias_87, buf238, buf241, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_87
        del getitem_79
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf238, buf241, 8388608, grid=grid(8388608), stream=stream0)
        buf243 = reinterpret_tensor(buf233, (8, 64, 1024), (65536, 1024, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_407, buf241, out=buf243)
        del permute_407
        buf248 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_11, buf243, buf248, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_11
        buf250 = reinterpret_tensor(buf243, (1024, 512), (512, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (1024, 512), (512, 1), 0), permute_418, out=buf250)
        del permute_418
        buf244 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf241, permute_408, out=buf244)
        del permute_408
        buf251 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf244, buf251, 524288, grid=grid(524288), stream=stream0)
        buf253 = reinterpret_tensor(buf244, (1024, 512), (512, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf251, permute_423, out=buf253)
        del permute_423
        buf256 = buf230; del buf230  # reuse
        buf257 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf256, buf247, buf250, buf253, primals_20, add_54, rsqrt_19, getitem_77, buf257, 1024, 512, grid=grid(1024), stream=stream0)
        del getitem_77
        del primals_20
        buf259 = reinterpret_tensor(buf198, (1024, 2048), (2048, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (1024, 512), (512, 1), 0), permute_427, out=buf259)
        del permute_427
        buf260 = reinterpret_tensor(buf259, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf259  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf260, le_5, getitem_75, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_75
        del le_5
        buf262 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (1024, 2048), (2048, 1), 0), permute_431, out=buf262)
        del permute_431
        buf265 = buf256; del buf256  # reuse
        buf266 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf265, buf262, primals_19, add_52, rsqrt_18, getitem_73, buf266, 1024, 512, grid=grid(1024), stream=stream0)
        del getitem_73
        del primals_19
        buf268 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (1024, 512), (512, 1), 0), permute_435, out=buf268)
        del permute_435
        buf269 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_438, reinterpret_tensor(buf268, (8, 1024, 64), (64, 512, 1), 0), out=buf269)
        del permute_438
        buf280 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_10, buf269, buf280, 524288, grid=grid(524288), stream=stream0)
        del tangents_10
        buf282 = reinterpret_tensor(buf269, (1024, 512), (512, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (1024, 512), (512, 1), 0), permute_446, out=buf282)
        del permute_446
        buf270 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf268, (8, 1024, 64), (64, 512, 1), 0), permute_439, out=buf270)
        del permute_439
        buf273 = reinterpret_tensor(buf235, (8388608, ), (1, ), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf273, 8388608, grid=grid(8388608), stream=stream0)
        buf276 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf270, getitem_71, alias_91, buf273, buf276, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_91
        del getitem_71
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf273, buf276, 8388608, grid=grid(8388608), stream=stream0)
        buf278 = reinterpret_tensor(buf268, (8, 64, 1024), (65536, 1024, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_440, buf276, out=buf278)
        del permute_440
        buf283 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_9, buf278, buf283, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_9
        buf285 = reinterpret_tensor(buf278, (1024, 512), (512, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (1024, 512), (512, 1), 0), permute_451, out=buf285)
        del permute_451
        buf279 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf276, permute_441, out=buf279)
        del permute_441
        buf286 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf279, buf286, 524288, grid=grid(524288), stream=stream0)
        buf288 = reinterpret_tensor(buf279, (1024, 512), (512, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf286, permute_456, out=buf288)
        del permute_456
        buf291 = buf265; del buf265  # reuse
        buf292 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf291, buf288, primals_18, add_49, rsqrt_17, getitem_69, buf292, 1024, 512, grid=grid(1024), stream=stream0)
        del getitem_69
        del primals_18
        buf294 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (1024, 512), (512, 1), 0), permute_460, out=buf294)
        del permute_460
        buf295 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_463, reinterpret_tensor(buf294, (8, 1024, 64), (64, 512, 1), 0), out=buf295)
        del permute_463
        buf306 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_8, buf295, buf306, 524288, grid=grid(524288), stream=stream0)
        del tangents_8
        buf308 = reinterpret_tensor(buf295, (1024, 512), (512, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf306, (1024, 512), (512, 1), 0), permute_471, out=buf308)
        del permute_471
        buf296 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (8, 1024, 64), (64, 512, 1), 0), permute_464, out=buf296)
        del permute_464
        buf299 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf299, 8388608, grid=grid(8388608), stream=stream0)
        buf302 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf296, getitem_67, alias_93, buf299, buf302, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_93
        del getitem_67
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf299, buf302, 8388608, grid=grid(8388608), stream=stream0)
        buf304 = reinterpret_tensor(buf294, (8, 64, 1024), (65536, 1024, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_465, buf302, out=buf304)
        del permute_465
        buf309 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_7, buf304, buf309, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_7
        buf311 = reinterpret_tensor(buf304, (1024, 512), (512, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (1024, 512), (512, 1), 0), permute_476, out=buf311)
        del permute_476
        buf305 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf302, permute_466, out=buf305)
        del permute_466
        buf312 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf305, buf312, 524288, grid=grid(524288), stream=stream0)
        buf314 = reinterpret_tensor(buf305, (1024, 512), (512, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf312, permute_481, out=buf314)
        del permute_481
        buf317 = buf291; del buf291  # reuse
        buf318 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf317, buf308, buf311, buf314, primals_17, add_46, rsqrt_16, getitem_65, buf318, 1024, 512, grid=grid(1024), stream=stream0)
        del getitem_65
        del primals_17
        buf320 = empty((1024, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf318, (1024, 512), (512, 1), 0), permute_485, out=buf320)
        del permute_485
        buf321 = reinterpret_tensor(buf320, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf320  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf321, le_6, getitem_63, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_63
        del le_6
        buf323 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (1024, 2048), (2048, 1), 0), permute_489, out=buf323)
        del permute_489
        buf326 = buf317; del buf317  # reuse
        buf327 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf326, buf323, primals_16, add_44, rsqrt_15, getitem_61, buf327, 1024, 512, grid=grid(1024), stream=stream0)
        del getitem_61
        del primals_16
        buf329 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (1024, 512), (512, 1), 0), permute_493, out=buf329)
        del permute_493
        buf330 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_496, reinterpret_tensor(buf329, (8, 1024, 64), (64, 512, 1), 0), out=buf330)
        del permute_496
        buf341 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_6, buf330, buf341, 524288, grid=grid(524288), stream=stream0)
        del tangents_6
        buf343 = reinterpret_tensor(buf330, (1024, 512), (512, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (1024, 512), (512, 1), 0), permute_504, out=buf343)
        del permute_504
        buf331 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf329, (8, 1024, 64), (64, 512, 1), 0), permute_497, out=buf331)
        del permute_497
        buf334 = reinterpret_tensor(buf296, (8388608, ), (1, ), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf334, 8388608, grid=grid(8388608), stream=stream0)
        buf337 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf331, getitem_59, alias_97, buf334, buf337, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_97
        del getitem_59
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf334, buf337, 8388608, grid=grid(8388608), stream=stream0)
        buf339 = reinterpret_tensor(buf329, (8, 64, 1024), (65536, 1024, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_498, buf337, out=buf339)
        del permute_498
        buf344 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_5, buf339, buf344, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_5
        buf346 = reinterpret_tensor(buf339, (1024, 512), (512, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (1024, 512), (512, 1), 0), permute_509, out=buf346)
        del permute_509
        buf224 = reinterpret_tensor(buf101, (1, 1024, 512), (524288, 512, 1), 0); del buf101  # reuse
        buf387 = buf224; del buf224  # reuse
        buf390 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        buf391 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_16.run(buf387, tangents_27, buf37, buf40, buf98, buf159, buf162, buf220, buf223, buf282, buf285, buf343, buf346, getitem_51, primals_13, add_33, rsqrt_12, getitem_49, buf390, buf391, 1024, 512, grid=grid(1024), stream=stream0)
        del buf159
        del buf162
        del buf220
        del buf223
        del buf282
        del buf285
        del buf343
        del buf346
        del buf37
        del buf40
        del buf98
        del getitem_49
        del getitem_51
        del primals_13
        del tangents_27
        buf226 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (512, 1024), (1, 512), 0), view_254, out=buf226)
        del buf225
        del view_254
        buf228 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_123], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf227, add_57, rsqrt_20, buf228, 512, 1024, grid=grid(512), stream=stream0)
        del add_57
        del buf227
        del rsqrt_20
        buf232 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (512, 1024), (1, 512), 0), view_252, out=buf232)
        del buf231
        del view_252
        buf246 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (512, 1024), (1, 512), 0), view_234, out=buf246)
        del buf245
        buf249 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (512, 1024), (1, 512), 0), view_234, out=buf249)
        del buf248
        buf252 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (512, 1024), (1, 512), 0), view_234, out=buf252)
        del buf251
        del view_234
        buf254 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_118], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf247, buf250, buf253, add_54, rsqrt_19, buf254, 512, 1024, grid=grid(512), stream=stream0)
        del add_54
        del buf247
        del buf250
        del buf253
        del rsqrt_19
        buf258 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (512, 1024), (1, 512), 0), view_232, out=buf258)
        del buf257
        del view_232
        buf261 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (2048, 1024), (1, 2048), 0), view_230, out=buf261)
        del buf260
        del view_230
        buf263 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_110], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf262, add_52, rsqrt_18, buf263, 512, 1024, grid=grid(512), stream=stream0)
        del add_52
        del buf262
        del rsqrt_18
        buf267 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (512, 1024), (1, 512), 0), view_228, out=buf267)
        del buf266
        del view_228
        buf281 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (512, 1024), (1, 512), 0), view_169, out=buf281)
        del buf280
        buf284 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (512, 1024), (1, 512), 0), view_169, out=buf284)
        del buf283
        buf287 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (512, 1024), (1, 512), 0), view_210, out=buf287)
        del buf286
        del view_210
        buf289 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_106], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf288, add_49, rsqrt_17, buf289, 512, 1024, grid=grid(512), stream=stream0)
        del add_49
        del buf288
        del rsqrt_17
        buf293 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (512, 1024), (1, 512), 0), view_208, out=buf293)
        del buf292
        del view_208
        buf307 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf306, (512, 1024), (1, 512), 0), view_190, out=buf307)
        del buf306
        buf310 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (512, 1024), (1, 512), 0), view_190, out=buf310)
        del buf309
        buf313 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (512, 1024), (1, 512), 0), view_190, out=buf313)
        del buf312
        del view_190
        buf315 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_101], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf308, buf311, buf314, add_46, rsqrt_16, buf315, 512, 1024, grid=grid(512), stream=stream0)
        del add_46
        del buf308
        del buf311
        del buf314
        del rsqrt_16
        buf319 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf318, (512, 1024), (1, 512), 0), view_188, out=buf319)
        del buf318
        del view_188
        buf322 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (2048, 1024), (1, 2048), 0), view_186, out=buf322)
        del view_186
        buf324 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf323, add_44, rsqrt_15, buf324, 512, 1024, grid=grid(512), stream=stream0)
        del add_44
        del rsqrt_15
        buf328 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (512, 1024), (1, 512), 0), view_184, out=buf328)
        del view_184
        buf340 = reinterpret_tensor(buf327, (8, 1024, 64), (65536, 64, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf337, permute_499, out=buf340)
        del permute_499
        buf342 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (512, 1024), (1, 512), 0), view_169, out=buf342)
        buf345 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (512, 1024), (1, 512), 0), view_169, out=buf345)
        del view_169
        buf347 = reinterpret_tensor(buf344, (1024, 512), (512, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf340, buf347, 524288, grid=grid(524288), stream=stream0)
        buf348 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (512, 1024), (1, 512), 0), view_166, out=buf348)
        del view_166
        buf349 = reinterpret_tensor(buf340, (1024, 512), (512, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf347, permute_514, out=buf349)
        del permute_514
        buf350 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_89], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf349, add_40, rsqrt_14, buf350, 512, 1024, grid=grid(512), stream=stream0)
        buf352 = buf326; del buf326  # reuse
        buf353 = reinterpret_tensor(buf347, (1, 1024, 512), (524288, 512, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf352, buf349, primals_15, add_40, rsqrt_14, getitem_57, buf353, 1024, 512, grid=grid(1024), stream=stream0)
        del add_40
        del getitem_57
        del primals_15
        del rsqrt_14
        buf354 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (512, 1024), (1, 512), 0), view_164, out=buf354)
        del view_164
        buf355 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (1024, 512), (512, 1), 0), permute_518, out=buf355)
        del permute_518
        buf356 = reinterpret_tensor(buf353, (8, 1024, 64), (65536, 64, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_521, reinterpret_tensor(buf355, (8, 1024, 64), (64, 512, 1), 0), out=buf356)
        del permute_521
        buf357 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf355, (8, 1024, 64), (64, 512, 1), 0), permute_522, out=buf357)
        del permute_522
        buf360 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf360, 8388608, grid=grid(8388608), stream=stream0)
        buf363 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf357, getitem_55, alias_99, buf360, buf363, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_99
        del getitem_55
        buf366 = reinterpret_tensor(buf357, (1024, 1024, 8), (1024, 1, 1048576), 0); del buf357  # reuse
        # Source Nodes: [loss], Original ATen: [aten.as_strided_scatter, aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_17.run(buf360, buf54, buf115, buf176, buf238, buf299, buf363, buf366, 8388608, grid=grid(8388608), stream=stream0)
        buf365 = empty((32, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_18.run(buf365, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf365, [add_37], buf366, True)
        del add_37
        buf369 = reinterpret_tensor(buf355, (8, 64, 1024), (65536, 1024, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_524, buf363, out=buf369)
        del permute_524
        buf370 = reinterpret_tensor(buf341, (8, 1024, 64), (65536, 64, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf363, permute_525, out=buf370)
        del permute_525
        buf371 = reinterpret_tensor(buf323, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_4, buf356, buf371, 524288, grid=grid(524288), stream=stream0)
        del tangents_4
        buf372 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (512, 1024), (1, 512), 0), view_146, out=buf372)
        buf373 = reinterpret_tensor(buf356, (1024, 512), (512, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (1024, 512), (512, 1), 0), permute_530, out=buf373)
        del permute_530
        buf374 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(tangents_3, buf369, buf374, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del tangents_3
        buf375 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (512, 1024), (1, 512), 0), view_146, out=buf375)
        buf376 = reinterpret_tensor(buf369, (1024, 512), (512, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (1024, 512), (512, 1), 0), permute_535, out=buf376)
        del permute_535
        buf377 = reinterpret_tensor(buf374, (1024, 512), (512, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf370, buf377, 524288, grid=grid(524288), stream=stream0)
        buf378 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (512, 1024), (1, 512), 0), view_146, out=buf378)
        del view_146
        buf379 = reinterpret_tensor(buf370, (1024, 512), (512, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf377, permute_540, out=buf379)
        del buf377
        del permute_540
        buf380 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf373, buf376, buf379, getitem_52, rsqrt_13, buf380, 512, 1024, grid=grid(512), stream=stream0)
        buf382 = buf352; del buf352  # reuse
        buf384 = buf382; del buf382  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum]
        triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_19.run(buf384, buf373, buf376, buf379, primals_14, getitem_52, rsqrt_13, getitem_53, view_145, 1024, 512, grid=grid(1024), stream=stream0)
        del buf373
        del buf376
        del getitem_52
        del getitem_53
        del primals_14
        del rsqrt_13
        buf383 = empty((32128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_20.run(buf383, 16449536, grid=grid(16449536), stream=stream0)
        aten.index_put_(buf383, [view_145], buf384, True)
        del view_145
        buf388 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_79], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf387, add_33, rsqrt_12, buf388, 512, 1024, grid=grid(512), stream=stream0)
        del add_33
        del rsqrt_12
        buf392 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (512, 1024), (1, 512), 0), view_143, out=buf392)
        del view_143
        buf393 = reinterpret_tensor(buf321, (1024, 2048), (2048, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (1024, 512), (512, 1), 0), permute_544, out=buf393)
        del permute_544
        buf394 = reinterpret_tensor(buf393, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf393  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf394, le_7, getitem_47, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_47
        del le_7
        buf395 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (2048, 1024), (1, 2048), 0), view_141, out=buf395)
        del view_141
        buf396 = reinterpret_tensor(buf391, (1024, 512), (512, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (1024, 2048), (2048, 1), 0), permute_548, out=buf396)
        del permute_548
        buf397 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_71], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf396, add_31, rsqrt_11, buf397, 512, 1024, grid=grid(512), stream=stream0)
        buf399 = buf390; del buf390  # reuse
        buf400 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf399, buf396, primals_12, add_31, rsqrt_11, getitem_45, buf400, 1024, 512, grid=grid(1024), stream=stream0)
        del add_31
        del getitem_45
        del primals_12
        del rsqrt_11
        buf401 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (512, 1024), (1, 512), 0), view_139, out=buf401)
        del view_139
        buf402 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (1024, 512), (512, 1), 0), permute_552, out=buf402)
        del permute_552
        buf403 = reinterpret_tensor(buf400, (8, 1024, 64), (65536, 64, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_555, reinterpret_tensor(buf402, (8, 1024, 64), (64, 512, 1), 0), out=buf403)
        del permute_555
        buf404 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf402, (8, 1024, 64), (64, 512, 1), 0), permute_556, out=buf404)
        del permute_556
        buf407 = reinterpret_tensor(buf366, (8388608, ), (1, ), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf407, 8388608, grid=grid(8388608), stream=stream0)
        buf410 = reinterpret_tensor(buf54, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf404, getitem_43, alias_104, buf407, buf410, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_104
        del getitem_43
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf407, buf410, 8388608, grid=grid(8388608), stream=stream0)
        buf412 = reinterpret_tensor(buf402, (8, 64, 1024), (65536, 1024, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_557, buf410, out=buf412)
        del permute_557
        buf413 = reinterpret_tensor(buf384, (8, 1024, 64), (65536, 64, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf410, permute_558, out=buf413)
        del permute_558
        buf414 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf403, buf414, 524288, grid=grid(524288), stream=stream0)
        buf415 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf414, (512, 1024), (1, 512), 0), view_121, out=buf415)
        buf416 = reinterpret_tensor(buf403, (1024, 512), (512, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf414, permute_563, out=buf416)
        del permute_563
        buf417 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (512, 1024), (1024, 1), 0), view_121, out=buf417)
        buf418 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (1024, 512), (1, 1024), 0), permute_568, out=buf418)
        del permute_568
        buf419 = reinterpret_tensor(buf412, (1024, 512), (512, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf413, buf419, 524288, grid=grid(524288), stream=stream0)
        buf420 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (512, 1024), (1, 512), 0), view_121, out=buf420)
        del view_121
        buf421 = reinterpret_tensor(buf413, (1024, 512), (512, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf419, permute_573, out=buf421)
        del permute_573
        buf422 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf416, buf418, buf421, add_28, rsqrt_10, buf422, 512, 1024, grid=grid(512), stream=stream0)
        buf424 = buf399; del buf399  # reuse
        buf425 = reinterpret_tensor(buf419, (1, 1024, 512), (524288, 512, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf424, buf416, buf418, buf421, primals_11, add_28, rsqrt_10, getitem_41, buf425, 1024, 512, grid=grid(1024), stream=stream0)
        del add_28
        del getitem_41
        del primals_11
        del rsqrt_10
        buf426 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (512, 1024), (1, 512), 0), view_119, out=buf426)
        del view_119
        buf427 = reinterpret_tensor(buf394, (1024, 2048), (2048, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (1024, 512), (512, 1), 0), permute_577, out=buf427)
        del permute_577
        buf428 = reinterpret_tensor(buf427, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf427  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf428, le_8, getitem_39, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_39
        del le_8
        buf429 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (2048, 1024), (1, 2048), 0), view_117, out=buf429)
        del view_117
        buf430 = reinterpret_tensor(buf425, (1024, 512), (512, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1024, 2048), (2048, 1), 0), permute_581, out=buf430)
        del permute_581
        buf431 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf430, add_26, rsqrt_9, buf431, 512, 1024, grid=grid(512), stream=stream0)
        buf433 = buf424; del buf424  # reuse
        buf434 = reinterpret_tensor(buf421, (1, 1024, 512), (524288, 512, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf433, buf430, primals_10, add_26, rsqrt_9, getitem_37, buf434, 1024, 512, grid=grid(1024), stream=stream0)
        del add_26
        del getitem_37
        del primals_10
        del rsqrt_9
        buf435 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (512, 1024), (1, 512), 0), view_115, out=buf435)
        del view_115
        buf436 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (1024, 512), (512, 1), 0), permute_585, out=buf436)
        del permute_585
        buf437 = reinterpret_tensor(buf434, (8, 1024, 64), (65536, 64, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_588, reinterpret_tensor(buf436, (8, 1024, 64), (64, 512, 1), 0), out=buf437)
        del permute_588
        buf438 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (8, 1024, 64), (64, 512, 1), 0), permute_589, out=buf438)
        del permute_589
        buf441 = reinterpret_tensor(buf404, (8388608, ), (1, ), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf441, 8388608, grid=grid(8388608), stream=stream0)
        buf444 = reinterpret_tensor(buf360, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf438, getitem_35, alias_108, buf441, buf444, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_108
        del getitem_35
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf441, buf444, 8388608, grid=grid(8388608), stream=stream0)
        buf446 = reinterpret_tensor(buf436, (8, 64, 1024), (65536, 1024, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_590, buf444, out=buf446)
        del permute_590
        buf447 = reinterpret_tensor(buf418, (8, 1024, 64), (65536, 64, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf444, permute_591, out=buf447)
        del permute_591
        buf448 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf437, buf448, 524288, grid=grid(524288), stream=stream0)
        buf449 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (512, 1024), (1, 512), 0), view_97, out=buf449)
        buf450 = reinterpret_tensor(buf437, (1024, 512), (512, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf448, permute_596, out=buf450)
        del permute_596
        buf451 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (512, 1024), (1024, 1), 0), view_97, out=buf451)
        buf452 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (1024, 512), (1, 1024), 0), permute_601, out=buf452)
        del permute_601
        buf453 = reinterpret_tensor(buf446, (1024, 512), (512, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf447, buf453, 524288, grid=grid(524288), stream=stream0)
        buf454 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (512, 1024), (1, 512), 0), view_97, out=buf454)
        del view_97
        buf455 = reinterpret_tensor(buf447, (1024, 512), (512, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf453, permute_606, out=buf455)
        del permute_606
        buf456 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_53], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf450, buf452, buf455, add_23, rsqrt_8, buf456, 512, 1024, grid=grid(512), stream=stream0)
        buf458 = buf433; del buf433  # reuse
        buf459 = reinterpret_tensor(buf453, (1, 1024, 512), (524288, 512, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf458, buf450, buf452, buf455, primals_9, add_23, rsqrt_8, getitem_33, buf459, 1024, 512, grid=grid(1024), stream=stream0)
        del add_23
        del getitem_33
        del primals_9
        del rsqrt_8
        buf460 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (512, 1024), (1, 512), 0), view_95, out=buf460)
        del view_95
        buf461 = reinterpret_tensor(buf428, (1024, 2048), (2048, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (1024, 512), (512, 1), 0), permute_610, out=buf461)
        del permute_610
        buf462 = reinterpret_tensor(buf461, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf461  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf462, le_9, getitem_31, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_31
        del le_9
        buf463 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (2048, 1024), (1, 2048), 0), view_93, out=buf463)
        del view_93
        buf464 = reinterpret_tensor(buf459, (1024, 512), (512, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (1024, 2048), (2048, 1), 0), permute_614, out=buf464)
        del permute_614
        buf465 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_45], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf464, add_21, rsqrt_7, buf465, 512, 1024, grid=grid(512), stream=stream0)
        buf467 = buf458; del buf458  # reuse
        buf468 = reinterpret_tensor(buf455, (1, 1024, 512), (524288, 512, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf467, buf464, primals_8, add_21, rsqrt_7, getitem_29, buf468, 1024, 512, grid=grid(1024), stream=stream0)
        del add_21
        del getitem_29
        del primals_8
        del rsqrt_7
        buf469 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (512, 1024), (1, 512), 0), view_91, out=buf469)
        del view_91
        buf470 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (1024, 512), (512, 1), 0), permute_618, out=buf470)
        del permute_618
        buf471 = reinterpret_tensor(buf468, (8, 1024, 64), (65536, 64, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_621, reinterpret_tensor(buf470, (8, 1024, 64), (64, 512, 1), 0), out=buf471)
        del permute_621
        buf472 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (8, 1024, 64), (64, 512, 1), 0), permute_622, out=buf472)
        del permute_622
        buf475 = reinterpret_tensor(buf438, (8388608, ), (1, ), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf475, 8388608, grid=grid(8388608), stream=stream0)
        buf478 = reinterpret_tensor(buf299, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf472, getitem_27, alias_112, buf475, buf478, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_112
        del getitem_27
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf475, buf478, 8388608, grid=grid(8388608), stream=stream0)
        buf480 = reinterpret_tensor(buf470, (8, 64, 1024), (65536, 1024, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_623, buf478, out=buf480)
        del permute_623
        buf481 = reinterpret_tensor(buf452, (8, 1024, 64), (65536, 64, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf478, permute_624, out=buf481)
        del permute_624
        buf482 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf471, buf482, 524288, grid=grid(524288), stream=stream0)
        buf483 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (512, 1024), (1, 512), 0), view_73, out=buf483)
        buf484 = reinterpret_tensor(buf471, (1024, 512), (512, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf482, permute_629, out=buf484)
        del permute_629
        buf485 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (512, 1024), (1024, 1), 0), view_73, out=buf485)
        buf486 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (1024, 512), (1, 1024), 0), permute_634, out=buf486)
        del permute_634
        buf487 = reinterpret_tensor(buf480, (1024, 512), (512, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf481, buf487, 524288, grid=grid(524288), stream=stream0)
        buf488 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (512, 1024), (1, 512), 0), view_73, out=buf488)
        del view_73
        buf489 = reinterpret_tensor(buf481, (1024, 512), (512, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf487, permute_639, out=buf489)
        del permute_639
        buf490 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_40], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf484, buf486, buf489, add_18, rsqrt_6, buf490, 512, 1024, grid=grid(512), stream=stream0)
        buf492 = buf467; del buf467  # reuse
        buf493 = reinterpret_tensor(buf487, (1, 1024, 512), (524288, 512, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf492, buf484, buf486, buf489, primals_7, add_18, rsqrt_6, getitem_25, buf493, 1024, 512, grid=grid(1024), stream=stream0)
        del add_18
        del getitem_25
        del primals_7
        del rsqrt_6
        buf494 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (512, 1024), (1, 512), 0), view_71, out=buf494)
        del view_71
        buf495 = reinterpret_tensor(buf462, (1024, 2048), (2048, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (1024, 512), (512, 1), 0), permute_643, out=buf495)
        del permute_643
        buf496 = reinterpret_tensor(buf495, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf495  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf496, le_10, getitem_23, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_23
        del le_10
        buf497 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (2048, 1024), (1, 2048), 0), view_69, out=buf497)
        del view_69
        buf498 = reinterpret_tensor(buf493, (1024, 512), (512, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (1024, 2048), (2048, 1), 0), permute_647, out=buf498)
        del permute_647
        buf499 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf498, add_16, rsqrt_5, buf499, 512, 1024, grid=grid(512), stream=stream0)
        buf501 = buf492; del buf492  # reuse
        buf502 = reinterpret_tensor(buf489, (1, 1024, 512), (524288, 512, 1), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf501, buf498, primals_6, add_16, rsqrt_5, getitem_21, buf502, 1024, 512, grid=grid(1024), stream=stream0)
        del add_16
        del getitem_21
        del primals_6
        del rsqrt_5
        buf503 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf502, (512, 1024), (1, 512), 0), view_67, out=buf503)
        del view_67
        buf504 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf502, (1024, 512), (512, 1), 0), permute_651, out=buf504)
        del permute_651
        buf505 = reinterpret_tensor(buf502, (8, 1024, 64), (65536, 64, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_654, reinterpret_tensor(buf504, (8, 1024, 64), (64, 512, 1), 0), out=buf505)
        del permute_654
        buf506 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf504, (8, 1024, 64), (64, 512, 1), 0), permute_655, out=buf506)
        del permute_655
        buf509 = reinterpret_tensor(buf472, (8388608, ), (1, ), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf509, 8388608, grid=grid(8388608), stream=stream0)
        buf512 = reinterpret_tensor(buf238, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf506, getitem_19, alias_116, buf509, buf512, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_116
        del getitem_19
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf509, buf512, 8388608, grid=grid(8388608), stream=stream0)
        buf514 = reinterpret_tensor(buf504, (8, 64, 1024), (65536, 1024, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_656, buf512, out=buf514)
        del permute_656
        buf515 = reinterpret_tensor(buf486, (8, 1024, 64), (65536, 64, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf512, permute_657, out=buf515)
        del permute_657
        buf516 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf505, buf516, 524288, grid=grid(524288), stream=stream0)
        buf517 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (512, 1024), (1, 512), 0), view_49, out=buf517)
        buf518 = reinterpret_tensor(buf505, (1024, 512), (512, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf516, permute_662, out=buf518)
        del permute_662
        buf519 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (512, 1024), (1024, 1), 0), view_49, out=buf519)
        buf520 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (1024, 512), (1, 1024), 0), permute_667, out=buf520)
        del permute_667
        buf521 = reinterpret_tensor(buf514, (1024, 512), (512, 1), 0); del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf515, buf521, 524288, grid=grid(524288), stream=stream0)
        buf522 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (512, 1024), (1, 512), 0), view_49, out=buf522)
        del view_49
        buf523 = reinterpret_tensor(buf515, (1024, 512), (512, 1), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf521, permute_672, out=buf523)
        del permute_672
        buf524 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf518, buf520, buf523, add_13, rsqrt_4, buf524, 512, 1024, grid=grid(512), stream=stream0)
        buf526 = buf501; del buf501  # reuse
        buf527 = reinterpret_tensor(buf521, (1, 1024, 512), (524288, 512, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf526, buf518, buf520, buf523, primals_5, add_13, rsqrt_4, getitem_17, buf527, 1024, 512, grid=grid(1024), stream=stream0)
        del add_13
        del getitem_17
        del primals_5
        del rsqrt_4
        buf528 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (512, 1024), (1, 512), 0), view_47, out=buf528)
        del view_47
        buf529 = reinterpret_tensor(buf496, (1024, 2048), (2048, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (1024, 512), (512, 1), 0), permute_676, out=buf529)
        del permute_676
        buf530 = reinterpret_tensor(buf529, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf529  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf530, le_11, getitem_15, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_15
        del le_11
        buf531 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (2048, 1024), (1, 2048), 0), view_45, out=buf531)
        del view_45
        buf532 = reinterpret_tensor(buf527, (1024, 512), (512, 1), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (1024, 2048), (2048, 1), 0), permute_680, out=buf532)
        del permute_680
        buf533 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_19], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf532, add_11, rsqrt_3, buf533, 512, 1024, grid=grid(512), stream=stream0)
        buf535 = buf526; del buf526  # reuse
        buf536 = reinterpret_tensor(buf523, (1, 1024, 512), (524288, 512, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf535, buf532, primals_4, add_11, rsqrt_3, getitem_13, buf536, 1024, 512, grid=grid(1024), stream=stream0)
        del add_11
        del getitem_13
        del primals_4
        del rsqrt_3
        buf537 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (512, 1024), (1, 512), 0), view_43, out=buf537)
        del view_43
        buf538 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (1024, 512), (512, 1), 0), permute_684, out=buf538)
        del permute_684
        buf539 = reinterpret_tensor(buf536, (8, 1024, 64), (65536, 64, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_687, reinterpret_tensor(buf538, (8, 1024, 64), (64, 512, 1), 0), out=buf539)
        del permute_687
        buf540 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf538, (8, 1024, 64), (64, 512, 1), 0), permute_688, out=buf540)
        del permute_688
        buf543 = reinterpret_tensor(buf506, (8388608, ), (1, ), 0); del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf543, 8388608, grid=grid(8388608), stream=stream0)
        buf546 = reinterpret_tensor(buf176, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf540, getitem_11, alias_120, buf543, buf546, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_120
        del getitem_11
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_10.run(buf543, buf546, 8388608, grid=grid(8388608), stream=stream0)
        buf548 = reinterpret_tensor(buf538, (8, 64, 1024), (65536, 1024, 1), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_689, buf546, out=buf548)
        del permute_689
        buf549 = reinterpret_tensor(buf520, (8, 1024, 64), (65536, 64, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf546, permute_690, out=buf549)
        del permute_690
        buf550 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf539, buf550, 524288, grid=grid(524288), stream=stream0)
        buf551 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf550, (512, 1024), (1, 512), 0), view_25, out=buf551)
        buf552 = reinterpret_tensor(buf539, (1024, 512), (512, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf550, permute_695, out=buf552)
        del permute_695
        buf553 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (512, 1024), (1024, 1), 0), view_25, out=buf553)
        buf554 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (1024, 512), (1, 1024), 0), permute_700, out=buf554)
        del permute_700
        buf555 = reinterpret_tensor(buf548, (1024, 512), (512, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf549, buf555, 524288, grid=grid(524288), stream=stream0)
        buf556 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (512, 1024), (1, 512), 0), view_25, out=buf556)
        del view_25
        buf557 = reinterpret_tensor(buf549, (1024, 512), (512, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf555, permute_705, out=buf557)
        del permute_705
        buf558 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf552, buf554, buf557, add_8, rsqrt_2, buf558, 512, 1024, grid=grid(512), stream=stream0)
        buf560 = buf535; del buf535  # reuse
        buf561 = reinterpret_tensor(buf555, (1, 1024, 512), (524288, 512, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15.run(buf560, buf552, buf554, buf557, primals_3, add_8, rsqrt_2, getitem_9, buf561, 1024, 512, grid=grid(1024), stream=stream0)
        del add_8
        del getitem_9
        del primals_3
        del rsqrt_2
        buf562 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf561, (512, 1024), (1, 512), 0), view_23, out=buf562)
        del view_23
        buf563 = reinterpret_tensor(buf530, (1024, 2048), (2048, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf561, (1024, 512), (512, 1), 0), permute_709, out=buf563)
        del permute_709
        buf564 = reinterpret_tensor(buf563, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf563  # reuse
        # Source Nodes: [loss], Original ATen: [aten.native_dropout_backward, aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_native_dropout_backward_nll_loss_forward_threshold_backward_5.run(buf564, le_12, getitem_7, 2097152, grid=grid(2097152), stream=stream0)
        del getitem_7
        del le_12
        buf565 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf564, (2048, 1024), (1, 2048), 0), view_21, out=buf565)
        del view_21
        buf566 = reinterpret_tensor(buf561, (1024, 512), (512, 1), 0); del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf564, (1024, 2048), (2048, 1), 0), permute_713, out=buf566)
        del buf564
        del permute_713
        buf567 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_6], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_6.run(buf566, add_6, rsqrt_1, buf567, 512, 1024, grid=grid(512), stream=stream0)
        buf569 = buf560; del buf560  # reuse
        buf570 = reinterpret_tensor(buf557, (1, 1024, 512), (524288, 512, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_7.run(buf569, buf566, primals_2, add_6, rsqrt_1, getitem_5, buf570, 1024, 512, grid=grid(1024), stream=stream0)
        del add_6
        del getitem_5
        del primals_2
        del rsqrt_1
        buf571 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf570, (512, 1024), (1, 512), 0), view_19, out=buf571)
        del view_19
        buf572 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf570, (1024, 512), (512, 1), 0), permute_717, out=buf572)
        del permute_717
        buf573 = reinterpret_tensor(buf570, (8, 1024, 64), (65536, 64, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_720, reinterpret_tensor(buf572, (8, 1024, 64), (64, 512, 1), 0), out=buf573)
        del permute_720
        buf574 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf572, (8, 1024, 64), (64, 512, 1), 0), permute_721, out=buf574)
        del permute_721
        buf577 = reinterpret_tensor(buf540, (8388608, ), (1, ), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_8.run(buf577, 8388608, grid=grid(8388608), stream=stream0)
        buf580 = reinterpret_tensor(buf115, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter, aten.native_dropout_backward, aten.squeeze]
        triton_per_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_9.run(buf574, getitem_3, alias_124, buf577, buf580, 8192, 1024, grid=grid(8192), stream=stream0)
        del alias_124
        del getitem_3
        buf583 = reinterpret_tensor(buf574, (1024, 1024, 8), (1024, 1, 1048576), 0); del buf574  # reuse
        # Source Nodes: [loss], Original ATen: [aten.as_strided_scatter, aten.embedding_dense_backward, aten.nll_loss_forward]
        triton_poi_fused_as_strided_scatter_embedding_dense_backward_nll_loss_forward_17.run(buf577, buf407, buf441, buf475, buf509, buf543, buf580, buf583, 8388608, grid=grid(8388608), stream=stream0)
        del buf407
        del buf441
        del buf475
        del buf509
        del buf543
        del buf577
        buf582 = empty((32, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_18.run(buf582, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf582, [add_3], buf583, True)
        del add_3
        del buf583
        buf586 = reinterpret_tensor(buf572, (8, 64, 1024), (65536, 1024, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_723, buf580, out=buf586)
        del permute_723
        buf587 = reinterpret_tensor(buf554, (8, 1024, 64), (65536, 64, 1), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf580, permute_724, out=buf587)
        del buf580
        del permute_724
        buf588 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf573, buf588, 524288, grid=grid(524288), stream=stream0)
        buf589 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (512, 1024), (1, 512), 0), view_1, out=buf589)
        buf590 = reinterpret_tensor(buf573, (1024, 512), (512, 1), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf588, permute_729, out=buf590)
        del permute_729
        buf591 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (512, 1024), (1024, 1), 0), view_1, out=buf591)
        buf592 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (1024, 512), (1, 1024), 0), permute_734, out=buf592)
        del permute_734
        buf593 = reinterpret_tensor(buf586, (1024, 512), (512, 1), 0); del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_13.run(buf587, buf593, 524288, grid=grid(524288), stream=stream0)
        buf594 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf593, (512, 1024), (1, 512), 0), view_1, out=buf594)
        del view_1
        buf595 = reinterpret_tensor(buf587, (1024, 512), (512, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf593, permute_739, out=buf595)
        del buf593
        del permute_739
        buf596 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_14.run(buf590, buf592, buf595, getitem, rsqrt, buf596, 512, 1024, grid=grid(512), stream=stream0)
        buf598 = buf569; del buf569  # reuse
        buf600 = buf598; del buf598  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum]
        triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_19.run(buf600, buf590, buf592, buf595, primals_1, getitem, rsqrt, getitem_1, view, 1024, 512, grid=grid(1024), stream=stream0)
        del buf590
        del buf592
        del buf595
        del getitem
        del getitem_1
        del primals_1
        del rsqrt
        buf599 = empty((32128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_20.run(buf599, 16449536, grid=grid(16449536), stream=stream0)
        aten.index_put_(buf599, [view], buf600, True)
        del buf600
        del view
        buf386 = empty((32128, 512), device='cuda', dtype=torch.float32)
        buf603 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf603, buf383, buf599, 16449536, grid=grid(16449536), stream=stream0)
        return (reinterpret_tensor(buf596, (512, ), (1, ), 0), reinterpret_tensor(buf567, (512, ), (1, ), 0), reinterpret_tensor(buf558, (512, ), (1, ), 0), reinterpret_tensor(buf533, (512, ), (1, ), 0), reinterpret_tensor(buf524, (512, ), (1, ), 0), reinterpret_tensor(buf499, (512, ), (1, ), 0), reinterpret_tensor(buf490, (512, ), (1, ), 0), reinterpret_tensor(buf465, (512, ), (1, ), 0), reinterpret_tensor(buf456, (512, ), (1, ), 0), reinterpret_tensor(buf431, (512, ), (1, ), 0), reinterpret_tensor(buf422, (512, ), (1, ), 0), reinterpret_tensor(buf397, (512, ), (1, ), 0), reinterpret_tensor(buf388, (512, ), (1, ), 0), reinterpret_tensor(buf380, (512, ), (1, ), 0), reinterpret_tensor(buf350, (512, ), (1, ), 0), reinterpret_tensor(buf324, (512, ), (1, ), 0), reinterpret_tensor(buf315, (512, ), (1, ), 0), reinterpret_tensor(buf289, (512, ), (1, ), 0), reinterpret_tensor(buf263, (512, ), (1, ), 0), reinterpret_tensor(buf254, (512, ), (1, ), 0), reinterpret_tensor(buf228, (512, ), (1, ), 0), reinterpret_tensor(buf201, (512, ), (1, ), 0), reinterpret_tensor(buf192, (512, ), (1, ), 0), reinterpret_tensor(buf166, (512, ), (1, ), 0), reinterpret_tensor(buf140, (512, ), (1, ), 0), reinterpret_tensor(buf131, (512, ), (1, ), 0), reinterpret_tensor(buf105, (512, ), (1, ), 0), reinterpret_tensor(buf79, (512, ), (1, ), 0), reinterpret_tensor(buf70, (512, ), (1, ), 0), reinterpret_tensor(buf44, (512, ), (1, ), 0), reinterpret_tensor(buf17, (512, ), (1, ), 0), reinterpret_tensor(buf8, (512, ), (1, ), 0), buf603, reinterpret_tensor(buf594, (512, 512), (512, 1), 0), reinterpret_tensor(buf591, (512, 512), (512, 1), 0), reinterpret_tensor(buf589, (512, 512), (512, 1), 0), buf582, reinterpret_tensor(buf571, (512, 512), (512, 1), 0), reinterpret_tensor(buf565, (2048, 512), (512, 1), 0), reinterpret_tensor(buf562, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf556, (512, 512), (512, 1), 0), reinterpret_tensor(buf553, (512, 512), (512, 1), 0), reinterpret_tensor(buf551, (512, 512), (512, 1), 0), reinterpret_tensor(buf537, (512, 512), (512, 1), 0), reinterpret_tensor(buf531, (2048, 512), (512, 1), 0), reinterpret_tensor(buf528, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf522, (512, 512), (512, 1), 0), reinterpret_tensor(buf519, (512, 512), (512, 1), 0), reinterpret_tensor(buf517, (512, 512), (512, 1), 0), reinterpret_tensor(buf503, (512, 512), (512, 1), 0), reinterpret_tensor(buf497, (2048, 512), (512, 1), 0), reinterpret_tensor(buf494, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf488, (512, 512), (512, 1), 0), reinterpret_tensor(buf485, (512, 512), (512, 1), 0), reinterpret_tensor(buf483, (512, 512), (512, 1), 0), reinterpret_tensor(buf469, (512, 512), (512, 1), 0), reinterpret_tensor(buf463, (2048, 512), (512, 1), 0), reinterpret_tensor(buf460, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf454, (512, 512), (512, 1), 0), reinterpret_tensor(buf451, (512, 512), (512, 1), 0), reinterpret_tensor(buf449, (512, 512), (512, 1), 0), reinterpret_tensor(buf435, (512, 512), (512, 1), 0), reinterpret_tensor(buf429, (2048, 512), (512, 1), 0), reinterpret_tensor(buf426, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf420, (512, 512), (512, 1), 0), reinterpret_tensor(buf417, (512, 512), (512, 1), 0), reinterpret_tensor(buf415, (512, 512), (512, 1), 0), reinterpret_tensor(buf401, (512, 512), (512, 1), 0), reinterpret_tensor(buf395, (2048, 512), (512, 1), 0), reinterpret_tensor(buf392, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf378, (512, 512), (512, 1), 0), reinterpret_tensor(buf375, (512, 512), (512, 1), 0), reinterpret_tensor(buf372, (512, 512), (512, 1), 0), buf365, reinterpret_tensor(buf354, (512, 512), (512, 1), 0), reinterpret_tensor(buf348, (512, 512), (512, 1), 0), reinterpret_tensor(buf345, (512, 512), (512, 1), 0), reinterpret_tensor(buf342, (512, 512), (512, 1), 0), reinterpret_tensor(buf328, (512, 512), (512, 1), 0), reinterpret_tensor(buf322, (2048, 512), (512, 1), 0), reinterpret_tensor(buf319, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf313, (512, 512), (512, 1), 0), reinterpret_tensor(buf310, (512, 512), (512, 1), 0), reinterpret_tensor(buf307, (512, 512), (512, 1), 0), reinterpret_tensor(buf293, (512, 512), (512, 1), 0), reinterpret_tensor(buf287, (512, 512), (512, 1), 0), reinterpret_tensor(buf284, (512, 512), (512, 1), 0), reinterpret_tensor(buf281, (512, 512), (512, 1), 0), reinterpret_tensor(buf267, (512, 512), (512, 1), 0), reinterpret_tensor(buf261, (2048, 512), (512, 1), 0), reinterpret_tensor(buf258, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf252, (512, 512), (512, 1), 0), reinterpret_tensor(buf249, (512, 512), (512, 1), 0), reinterpret_tensor(buf246, (512, 512), (512, 1), 0), reinterpret_tensor(buf232, (512, 512), (512, 1), 0), reinterpret_tensor(buf226, (512, 512), (512, 1), 0), reinterpret_tensor(buf222, (512, 512), (512, 1), 0), reinterpret_tensor(buf219, (512, 512), (512, 1), 0), reinterpret_tensor(buf205, (512, 512), (512, 1), 0), reinterpret_tensor(buf199, (2048, 512), (512, 1), 0), reinterpret_tensor(buf196, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf190, (512, 512), (512, 1), 0), reinterpret_tensor(buf187, (512, 512), (512, 1), 0), reinterpret_tensor(buf184, (512, 512), (512, 1), 0), reinterpret_tensor(buf170, (512, 512), (512, 1), 0), reinterpret_tensor(buf164, (512, 512), (512, 1), 0), reinterpret_tensor(buf161, (512, 512), (512, 1), 0), reinterpret_tensor(buf158, (512, 512), (512, 1), 0), reinterpret_tensor(buf144, (512, 512), (512, 1), 0), reinterpret_tensor(buf138, (2048, 512), (512, 1), 0), reinterpret_tensor(buf135, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf129, (512, 512), (512, 1), 0), reinterpret_tensor(buf126, (512, 512), (512, 1), 0), reinterpret_tensor(buf123, (512, 512), (512, 1), 0), reinterpret_tensor(buf109, (512, 512), (512, 1), 0), reinterpret_tensor(buf103, (512, 512), (512, 1), 0), reinterpret_tensor(buf100, (512, 512), (512, 1), 0), reinterpret_tensor(buf97, (512, 512), (512, 1), 0), reinterpret_tensor(buf83, (512, 512), (512, 1), 0), reinterpret_tensor(buf77, (2048, 512), (512, 1), 0), reinterpret_tensor(buf74, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf68, (512, 512), (512, 1), 0), reinterpret_tensor(buf65, (512, 512), (512, 1), 0), reinterpret_tensor(buf62, (512, 512), (512, 1), 0), reinterpret_tensor(buf48, (512, 512), (512, 1), 0), reinterpret_tensor(buf42, (512, 512), (512, 1), 0), reinterpret_tensor(buf39, (512, 512), (512, 1), 0), reinterpret_tensor(buf36, (512, 512), (512, 1), 0), reinterpret_tensor(buf21, (512, 512), (512, 1), 0), reinterpret_tensor(buf15, (2048, 512), (512, 1), 0), reinterpret_tensor(buf12, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf6, (32128, 512), (512, 1), 0), None, None, None, )


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
    primals_134 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    getitem = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    rsqrt = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_3 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    getitem_3 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_19 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_6 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_23 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_8 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_43 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_11 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_47 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_13 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_4 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_67 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_16 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_23 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_71 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_18 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_91 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_21 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_95 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_33 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_23 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_8 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_115 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_26 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_119 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_28 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_139 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_45 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_31 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_143 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_33 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_12 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    view_145 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    getitem_52 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    rsqrt_13 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_37 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    getitem_55 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_164 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_40 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_14 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_169 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_184 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_44 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_186 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_63 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_188 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_65 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_46 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_208 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_49 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_17 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_210 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_228 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_52 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_75 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_232 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_54 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_234 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_252 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_57 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_254 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_272 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_60 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_21 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_276 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_89 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_62 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_22 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_278 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_296 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_93 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_65 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_23 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_298 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_316 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_68 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_24 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_318 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_99 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_320 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_70 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_322 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_340 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_73 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_26 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_342 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_360 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_76 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_27 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_362 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_364 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_78 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_28 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_366 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_115 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_384 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_81 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_29 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_386 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_119 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_404 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_84 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_30 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_406 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    view_408 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    add_86 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_31 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.bool)
    view_410 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    sub_24 = rand_strided((1024, 32128), (32128, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_7 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_199 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_67 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_69 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_239 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_257 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_73 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_272 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_75 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_292 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_297 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_302 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_315 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_79 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_330 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_347 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_81 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_373 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_380 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_85 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_87 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_407 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_423 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_431 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_91 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_441 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_451 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_456 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_93 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_466 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_489 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_493 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_97 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_499 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_504 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_99 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_524 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_540 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_544 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_548 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_552 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_556 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_104 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_557 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_563 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_568 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_573 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_577 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_581 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_585 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_588 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_589 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_108 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_596 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_601 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_606 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_610 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_614 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_618 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_621 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_622 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_112 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_623 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_629 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_634 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_639 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_643 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_10 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_647 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_651 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_654 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_655 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_116 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_656 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_657 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_662 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_667 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_672 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_676 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_680 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_684 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_688 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_120 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_689 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_690 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_695 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_700 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_705 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_709 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_713 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_720 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    alias_124 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_723 = rand_strided((8, 64, 1024), (64, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_724 = rand_strided((8, 1024, 64), (64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_734 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_739 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 1024, 32128), (32899072, 32128, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_134, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, getitem_7, view_23, getitem_9, add_8, rsqrt_2, view_25, getitem_11, view_43, getitem_13, add_11, rsqrt_3, view_45, getitem_15, view_47, getitem_17, add_13, rsqrt_4, view_49, getitem_19, view_67, getitem_21, add_16, rsqrt_5, view_69, getitem_23, view_71, getitem_25, add_18, rsqrt_6, view_73, getitem_27, view_91, getitem_29, add_21, rsqrt_7, view_93, getitem_31, view_95, getitem_33, add_23, rsqrt_8, view_97, getitem_35, view_115, getitem_37, add_26, rsqrt_9, view_117, getitem_39, view_119, getitem_41, add_28, rsqrt_10, view_121, getitem_43, view_139, getitem_45, add_31, rsqrt_11, view_141, getitem_47, view_143, getitem_49, add_33, rsqrt_12, getitem_51, view_145, getitem_52, getitem_53, rsqrt_13, view_146, add_37, getitem_55, view_164, getitem_57, add_40, rsqrt_14, view_166, view_169, getitem_59, view_184, getitem_61, add_44, rsqrt_15, view_186, getitem_63, view_188, getitem_65, add_46, rsqrt_16, view_190, getitem_67, view_208, getitem_69, add_49, rsqrt_17, view_210, getitem_71, view_228, getitem_73, add_52, rsqrt_18, view_230, getitem_75, view_232, getitem_77, add_54, rsqrt_19, view_234, getitem_79, view_252, getitem_81, add_57, rsqrt_20, view_254, getitem_83, view_272, getitem_85, add_60, rsqrt_21, view_274, getitem_87, view_276, getitem_89, add_62, rsqrt_22, view_278, getitem_91, view_296, getitem_93, add_65, rsqrt_23, view_298, getitem_95, view_316, getitem_97, add_68, rsqrt_24, view_318, getitem_99, view_320, getitem_101, add_70, rsqrt_25, view_322, getitem_103, view_340, getitem_105, add_73, rsqrt_26, view_342, getitem_107, view_360, getitem_109, add_76, rsqrt_27, view_362, getitem_111, view_364, getitem_113, add_78, rsqrt_28, view_366, getitem_115, view_384, getitem_117, add_81, rsqrt_29, view_386, getitem_119, view_404, getitem_121, add_84, rsqrt_30, view_406, getitem_123, view_408, getitem_125, add_86, rsqrt_31, getitem_127, view_410, sub_24, convert_element_type_7, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_67, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_69, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_73, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_75, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_79, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_81, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_85, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_87, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_91, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_93, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_97, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_99, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_104, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_108, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_112, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_116, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_120, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_124, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('T5ForConditionalGeneration', benchmark_compiled_module)
