
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


# kernel path: /tmp/torchinductor_youkaichao/jn/cjn6f2hn4432x5rhepdlprilugmw5fma5zqzihufyvz5h443vvdo.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384000
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


# kernel path: /tmp/torchinductor_youkaichao/ji/cjivva35nrjxummuzug4vgfwcnmnn5qo5bkjhdvca6ghm43la35h.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
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
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clchnoe3fron4od7fl56ynorv2yzhgtgfzrp5qea6aoicmgmc5cv.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp15 = tl.load(in_ptr4 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr1 + (r1 + (32000*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jz/cjz2s4ms7saklb32tw6y74qpaytdd5cb4u4r2wku65gnpuhfmk7r.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32000
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32000*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47rtllowhfh32hszespy3mvjpgweihbr5o2pcss4kxpyi3ml32j.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp7 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = 1024.0
    tmp20 = tmp7 * tmp19
    tmp21 = tmp20 - tmp11
    tmp22 = tmp12 * tmp17
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp3
    tmp28 = tmp24 * tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp24, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp28, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsr2whyq5o6ltko74u6tzousznwu5vxnmdkd3l2q7bwyyeqowsm.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnaeswmp3uqzn7xwlycneyhxluabiwieeryuwpvnsedubwgyz2e.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dv4lgev253toikcyfs5sbbcoc7oyvqtu5srkfrcxgwi55qqkm5.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/ces25bzppf76zsd6esdf75iroajvzkxuiq43252ct25zqz3f4sgl.py
# Source Nodes: [output_187], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
# output_187 => add_262, erf_23, mul_192
triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.int1)
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = 0.7071067811865476
    tmp8 = tmp6 * tmp7
    tmp9 = tl.math.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 * tmp6
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tl.exp(tmp16)
    tmp18 = 0.3989422804014327
    tmp19 = tmp17 * tmp18
    tmp20 = tmp6 * tmp19
    tmp21 = tmp13 + tmp20
    tmp22 = tmp5 * tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hz/chzmyhp5ukh3qpi2jahv57afjf6nfaek2p3rxfw245z7junppwso.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (524288*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/cojjvxmw2abyi5m6ipyaowy6lbscfjcyjttr6lnkenxewttl2eg5.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrap65gevpggdo44q774a4jwh4tq2mfqrickxmmjb3ureqkc5cw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 1024.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxam727cvqp6tms6y2l6w3bj3tsgnvpkyptgq27hfhzaovsfe2rg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gkyrdvetpzs5657mb7yweshmywnjjkleg4p37ayavd7qra27xz.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
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
    tmp14 = 0.125
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4h/c4h6syidihud4zo63g6x7ydyfin7d6um3lqq5jwj577xdnsuaf5c.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]

triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8380416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cnowunlcgxd76yrhmrkebtn3fgnhtwsjuizyqsb7bazyhd3cgd6i.py
# Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]

triton_poi_fused_bmm_slice_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_slice_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 1024
    x2 = (xindex // 524288)
    x3 = xindex % 524288
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-512) + x3 + (523776*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr1 + (x4), tmp7, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvp73dmusi5tceveltt2gycqeufi5v476ve5vjhpdvg2m6zrmker.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (32768*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kakjt35uo2sefc4jomem3hiwztr35jljkr6c6asejaygg7bdck.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (32768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wpdmm47imw7562fd7yu3yil4cy6yfaec2rfxzq323ezqcxqlxu.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
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
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (1024*x1)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4aqkoackt22pirzd3k4nkyeduk6rre43hnow5ssk3blx6bs3jxl.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 1024.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvl2drlex23aqfh5alh4mwht2dsw46x4bdts56y65xyvjv4d5cnz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjdl2nlr2vb7zv6s6eh5xtnfnj57qzpxnuxyzk3gpw7hstpylaa.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjkkuqqh676rlze65hiyyqdf6eqdtijue4wk6pnlavqamt4ih4b5.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_nll_loss_forward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_nll_loss_forward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp8 = tl.load(in_ptr3 + (x2), None)
    tmp10 = tl.load(in_ptr4 + (x2), None).to(tl.int1)
    tmp1 = tl.full([1], -1, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 * tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp15, tmp14)
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_170, primals_176, primals_178, primals_184, primals_186, primals_192, primals_194, primals_200, primals_202, primals_208, primals_210, primals_216, primals_218, primals_224, primals_226, primals_232, primals_234, primals_240, primals_242, primals_248, primals_250, primals_256, primals_258, primals_264, primals_266, primals_272, primals_274, primals_280, primals_282, primals_288, primals_290, primals_296, primals_298, primals_304, primals_306, primals_312, primals_314, primals_320, primals_322, primals_328, primals_330, primals_336, primals_338, primals_344, primals_346, primals_352, primals_354, primals_360, primals_365, permute, getitem_1, iota_2, getitem_5, getitem_7, mul_5, view_34, addmm, getitem_11, view_36, getitem_13, mul_10, getitem_17, getitem_19, mul_13, view_72, addmm_2, getitem_23, view_74, getitem_25, mul_18, getitem_29, getitem_31, mul_21, view_110, addmm_4, getitem_35, view_112, getitem_37, mul_26, getitem_41, getitem_43, mul_29, view_148, addmm_6, getitem_47, view_150, getitem_49, mul_34, getitem_53, getitem_55, mul_37, view_186, addmm_8, getitem_59, view_188, getitem_61, mul_42, getitem_65, getitem_67, mul_45, view_224, addmm_10, getitem_71, view_226, getitem_73, mul_50, getitem_77, getitem_79, mul_53, view_262, addmm_12, getitem_83, view_264, getitem_85, mul_58, getitem_89, getitem_91, mul_61, view_300, addmm_14, getitem_95, view_302, getitem_97, mul_66, getitem_101, getitem_103, mul_69, view_338, addmm_16, getitem_107, view_340, getitem_109, mul_74, getitem_113, getitem_115, mul_77, view_376, addmm_18, getitem_119, view_378, getitem_121, mul_82, getitem_125, getitem_127, mul_85, view_414, addmm_20, getitem_131, view_416, getitem_133, mul_90, getitem_137, getitem_139, mul_93, view_452, addmm_22, getitem_143, view_454, getitem_145, mul_98, getitem_149, getitem_151, mul_101, view_490, addmm_24, getitem_155, view_492, getitem_157, mul_106, getitem_161, getitem_163, mul_109, view_528, addmm_26, getitem_167, view_530, getitem_169, mul_114, getitem_173, getitem_175, mul_117, view_566, addmm_28, getitem_179, view_568, getitem_181, mul_122, getitem_185, getitem_187, mul_125, view_604, addmm_30, getitem_191, view_606, getitem_193, mul_130, getitem_197, getitem_199, mul_133, view_642, addmm_32, getitem_203, view_644, getitem_205, mul_138, getitem_209, getitem_211, mul_141, view_680, addmm_34, getitem_215, view_682, getitem_217, mul_146, getitem_221, getitem_223, mul_149, view_718, addmm_36, getitem_227, view_720, getitem_229, mul_154, getitem_233, getitem_235, mul_157, view_756, addmm_38, getitem_239, view_758, getitem_241, mul_162, getitem_245, getitem_247, mul_165, view_794, addmm_40, getitem_251, view_796, getitem_253, mul_170, getitem_257, getitem_259, mul_173, view_832, addmm_42, getitem_263, view_834, getitem_265, mul_178, getitem_269, getitem_271, mul_181, view_870, addmm_44, getitem_275, view_872, getitem_277, mul_186, getitem_281, getitem_283, mul_189, view_908, addmm_46, getitem_287, view_910, getitem_289, mul_194, getitem_293, view_912, sub_73, convert_element_type_5, permute_1013, div_27, permute_1018, permute_1022, div_28, permute_1027, permute_1028, permute_1034, permute_1035, alias_26, permute_1041, permute_1042, permute_1048, permute_1049, permute_1055, permute_1059, permute_1060, permute_1067, permute_1074, div_29, permute_1079, permute_1083, div_30, permute_1088, permute_1089, permute_1095, permute_1096, alias_27, permute_1102, permute_1103, permute_1109, permute_1110, permute_1120, permute_1121, permute_1128, permute_1135, div_31, permute_1140, permute_1144, div_32, permute_1149, permute_1150, permute_1156, permute_1157, alias_28, permute_1163, permute_1164, permute_1170, permute_1171, permute_1181, permute_1182, permute_1189, permute_1196, div_33, permute_1201, permute_1205, div_34, permute_1210, permute_1211, permute_1217, permute_1218, alias_29, permute_1224, permute_1225, permute_1231, permute_1232, permute_1242, permute_1243, permute_1250, permute_1257, div_35, permute_1262, permute_1266, div_36, permute_1271, permute_1272, permute_1278, permute_1279, alias_30, permute_1285, permute_1286, permute_1292, permute_1293, permute_1303, permute_1304, permute_1311, permute_1318, div_37, permute_1323, permute_1327, div_38, permute_1332, permute_1333, permute_1339, permute_1340, alias_31, permute_1346, permute_1347, permute_1353, permute_1354, permute_1364, permute_1365, permute_1372, permute_1379, div_39, permute_1384, permute_1388, div_40, permute_1393, permute_1394, permute_1400, permute_1401, alias_32, permute_1407, permute_1408, permute_1414, permute_1415, permute_1425, permute_1426, permute_1433, permute_1440, div_41, permute_1445, permute_1449, div_42, permute_1454, permute_1455, permute_1461, permute_1462, alias_33, permute_1468, permute_1469, permute_1475, permute_1476, permute_1486, permute_1487, permute_1494, permute_1501, div_43, permute_1506, permute_1510, div_44, permute_1515, permute_1516, permute_1522, permute_1523, alias_34, permute_1529, permute_1530, permute_1536, permute_1537, permute_1547, permute_1548, permute_1555, permute_1562, div_45, permute_1567, permute_1571, div_46, permute_1576, permute_1577, permute_1583, permute_1584, alias_35, permute_1590, permute_1591, permute_1597, permute_1598, permute_1608, permute_1609, permute_1616, permute_1623, div_47, permute_1628, permute_1632, div_48, permute_1637, permute_1638, permute_1644, permute_1645, alias_36, permute_1651, permute_1652, permute_1658, permute_1659, permute_1669, permute_1670, permute_1677, permute_1684, div_49, permute_1689, permute_1693, div_50, permute_1698, permute_1699, permute_1705, permute_1706, alias_37, permute_1712, permute_1713, permute_1719, permute_1720, permute_1730, permute_1731, permute_1738, permute_1745, div_51, permute_1750, permute_1754, div_52, permute_1759, permute_1760, permute_1766, permute_1767, alias_38, permute_1773, permute_1774, permute_1780, permute_1781, permute_1791, permute_1792, permute_1799, permute_1806, div_53, permute_1811, permute_1815, div_54, permute_1820, permute_1821, permute_1827, permute_1828, alias_39, permute_1834, permute_1835, permute_1841, permute_1842, permute_1852, permute_1853, permute_1860, permute_1867, div_55, permute_1872, permute_1876, div_56, permute_1881, permute_1882, permute_1888, permute_1889, alias_40, permute_1895, permute_1896, permute_1902, permute_1903, permute_1913, permute_1914, permute_1921, permute_1928, div_57, permute_1933, permute_1937, div_58, permute_1942, permute_1943, permute_1949, permute_1950, alias_41, permute_1956, permute_1957, permute_1963, permute_1964, permute_1974, permute_1975, permute_1982, permute_1989, div_59, permute_1994, permute_1998, div_60, permute_2003, permute_2004, permute_2010, permute_2011, alias_42, permute_2017, permute_2018, permute_2024, permute_2025, permute_2035, permute_2036, permute_2043, permute_2050, div_61, permute_2055, permute_2059, div_62, permute_2064, permute_2065, permute_2071, permute_2072, alias_43, permute_2078, permute_2079, permute_2085, permute_2086, permute_2096, permute_2097, permute_2104, permute_2111, div_63, permute_2116, permute_2120, div_64, permute_2125, permute_2126, permute_2132, permute_2133, alias_44, permute_2139, permute_2140, permute_2146, permute_2147, permute_2157, permute_2158, permute_2165, permute_2172, div_65, permute_2177, permute_2181, div_66, permute_2186, permute_2187, permute_2193, permute_2194, alias_45, permute_2200, permute_2201, permute_2207, permute_2208, permute_2218, permute_2219, permute_2226, permute_2233, div_67, permute_2238, permute_2242, div_68, permute_2247, permute_2248, permute_2254, permute_2255, alias_46, permute_2261, permute_2262, permute_2268, permute_2269, permute_2279, permute_2280, permute_2287, permute_2294, div_69, permute_2299, permute_2303, div_70, permute_2308, permute_2309, permute_2315, permute_2316, alias_47, permute_2322, permute_2323, permute_2329, permute_2330, permute_2340, permute_2341, permute_2348, permute_2355, div_71, permute_2360, permute_2364, div_72, permute_2369, permute_2370, permute_2376, permute_2377, alias_48, permute_2383, permute_2384, permute_2390, permute_2391, permute_2401, permute_2402, permute_2409, permute_2416, div_73, permute_2421, permute_2425, div_74, permute_2430, permute_2431, permute_2437, permute_2438, alias_49, permute_2444, permute_2445, permute_2451, permute_2452, permute_2462, permute_2463, permute_2470, permute_2477, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_170, (1024, ), (1, ))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_178, (1024, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, ), (1, ))
    assert_size_stride(primals_192, (1024, ), (1, ))
    assert_size_stride(primals_194, (1024, ), (1, ))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_202, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, ), (1, ))
    assert_size_stride(primals_210, (1024, ), (1, ))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_224, (1024, ), (1, ))
    assert_size_stride(primals_226, (1024, ), (1, ))
    assert_size_stride(primals_232, (1024, ), (1, ))
    assert_size_stride(primals_234, (1024, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_248, (1024, ), (1, ))
    assert_size_stride(primals_250, (1024, ), (1, ))
    assert_size_stride(primals_256, (1024, ), (1, ))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_264, (1024, ), (1, ))
    assert_size_stride(primals_266, (1024, ), (1, ))
    assert_size_stride(primals_272, (1024, ), (1, ))
    assert_size_stride(primals_274, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_290, (1024, ), (1, ))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_298, (1024, ), (1, ))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_322, (1024, ), (1, ))
    assert_size_stride(primals_328, (1024, ), (1, ))
    assert_size_stride(primals_330, (1024, ), (1, ))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_338, (1024, ), (1, ))
    assert_size_stride(primals_344, (1024, ), (1, ))
    assert_size_stride(primals_346, (1024, ), (1, ))
    assert_size_stride(primals_352, (1024, ), (1, ))
    assert_size_stride(primals_354, (1024, ), (1, ))
    assert_size_stride(primals_360, (1024, ), (1, ))
    assert_size_stride(primals_365, (1, 512), (512, 1))
    assert_size_stride(permute, (512, 1), (1, 512))
    assert_size_stride(getitem_1, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(iota_2, (512, ), (1, ))
    assert_size_stride(getitem_5, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_7, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_5, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_34, (512, 1024), (1024, 1))
    assert_size_stride(addmm, (512, 4096), (4096, 1))
    assert_size_stride(getitem_11, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_36, (512, 4096), (4096, 1))
    assert_size_stride(getitem_13, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_10, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_17, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_19, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_13, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_72, (512, 1024), (1024, 1))
    assert_size_stride(addmm_2, (512, 4096), (4096, 1))
    assert_size_stride(getitem_23, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_74, (512, 4096), (4096, 1))
    assert_size_stride(getitem_25, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_18, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_29, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_31, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_21, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_110, (512, 1024), (1024, 1))
    assert_size_stride(addmm_4, (512, 4096), (4096, 1))
    assert_size_stride(getitem_35, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_112, (512, 4096), (4096, 1))
    assert_size_stride(getitem_37, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_26, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_41, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_43, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_29, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_148, (512, 1024), (1024, 1))
    assert_size_stride(addmm_6, (512, 4096), (4096, 1))
    assert_size_stride(getitem_47, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_150, (512, 4096), (4096, 1))
    assert_size_stride(getitem_49, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_34, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_53, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_55, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_37, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_186, (512, 1024), (1024, 1))
    assert_size_stride(addmm_8, (512, 4096), (4096, 1))
    assert_size_stride(getitem_59, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_188, (512, 4096), (4096, 1))
    assert_size_stride(getitem_61, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_42, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_65, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_67, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_45, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_224, (512, 1024), (1024, 1))
    assert_size_stride(addmm_10, (512, 4096), (4096, 1))
    assert_size_stride(getitem_71, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_226, (512, 4096), (4096, 1))
    assert_size_stride(getitem_73, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_50, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_77, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_79, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_53, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_262, (512, 1024), (1024, 1))
    assert_size_stride(addmm_12, (512, 4096), (4096, 1))
    assert_size_stride(getitem_83, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_264, (512, 4096), (4096, 1))
    assert_size_stride(getitem_85, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_58, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_89, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_91, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_61, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_300, (512, 1024), (1024, 1))
    assert_size_stride(addmm_14, (512, 4096), (4096, 1))
    assert_size_stride(getitem_95, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_302, (512, 4096), (4096, 1))
    assert_size_stride(getitem_97, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_66, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_101, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_103, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_69, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_338, (512, 1024), (1024, 1))
    assert_size_stride(addmm_16, (512, 4096), (4096, 1))
    assert_size_stride(getitem_107, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_340, (512, 4096), (4096, 1))
    assert_size_stride(getitem_109, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_74, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_113, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_115, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_77, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_376, (512, 1024), (1024, 1))
    assert_size_stride(addmm_18, (512, 4096), (4096, 1))
    assert_size_stride(getitem_119, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_378, (512, 4096), (4096, 1))
    assert_size_stride(getitem_121, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_82, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_125, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_127, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_85, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_414, (512, 1024), (1024, 1))
    assert_size_stride(addmm_20, (512, 4096), (4096, 1))
    assert_size_stride(getitem_131, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_416, (512, 4096), (4096, 1))
    assert_size_stride(getitem_133, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_90, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_137, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_139, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_93, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_452, (512, 1024), (1024, 1))
    assert_size_stride(addmm_22, (512, 4096), (4096, 1))
    assert_size_stride(getitem_143, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_454, (512, 4096), (4096, 1))
    assert_size_stride(getitem_145, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_98, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_149, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_151, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_101, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_490, (512, 1024), (1024, 1))
    assert_size_stride(addmm_24, (512, 4096), (4096, 1))
    assert_size_stride(getitem_155, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_492, (512, 4096), (4096, 1))
    assert_size_stride(getitem_157, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_106, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_161, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_163, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_109, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_528, (512, 1024), (1024, 1))
    assert_size_stride(addmm_26, (512, 4096), (4096, 1))
    assert_size_stride(getitem_167, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_530, (512, 4096), (4096, 1))
    assert_size_stride(getitem_169, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_114, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_173, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_175, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_117, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_566, (512, 1024), (1024, 1))
    assert_size_stride(addmm_28, (512, 4096), (4096, 1))
    assert_size_stride(getitem_179, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_568, (512, 4096), (4096, 1))
    assert_size_stride(getitem_181, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_122, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_185, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_187, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_125, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_604, (512, 1024), (1024, 1))
    assert_size_stride(addmm_30, (512, 4096), (4096, 1))
    assert_size_stride(getitem_191, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_606, (512, 4096), (4096, 1))
    assert_size_stride(getitem_193, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_130, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_197, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_199, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_133, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_642, (512, 1024), (1024, 1))
    assert_size_stride(addmm_32, (512, 4096), (4096, 1))
    assert_size_stride(getitem_203, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_644, (512, 4096), (4096, 1))
    assert_size_stride(getitem_205, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_138, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_209, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_211, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_141, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_680, (512, 1024), (1024, 1))
    assert_size_stride(addmm_34, (512, 4096), (4096, 1))
    assert_size_stride(getitem_215, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_682, (512, 4096), (4096, 1))
    assert_size_stride(getitem_217, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_146, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_221, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_223, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_149, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_718, (512, 1024), (1024, 1))
    assert_size_stride(addmm_36, (512, 4096), (4096, 1))
    assert_size_stride(getitem_227, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_720, (512, 4096), (4096, 1))
    assert_size_stride(getitem_229, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_154, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_233, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_235, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_157, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_756, (512, 1024), (1024, 1))
    assert_size_stride(addmm_38, (512, 4096), (4096, 1))
    assert_size_stride(getitem_239, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_758, (512, 4096), (4096, 1))
    assert_size_stride(getitem_241, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_162, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_245, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_247, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_165, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_794, (512, 1024), (1024, 1))
    assert_size_stride(addmm_40, (512, 4096), (4096, 1))
    assert_size_stride(getitem_251, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_796, (512, 4096), (4096, 1))
    assert_size_stride(getitem_253, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_170, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_257, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_259, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_173, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_832, (512, 1024), (1024, 1))
    assert_size_stride(addmm_42, (512, 4096), (4096, 1))
    assert_size_stride(getitem_263, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_834, (512, 4096), (4096, 1))
    assert_size_stride(getitem_265, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_178, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_269, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_271, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_181, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_870, (512, 1024), (1024, 1))
    assert_size_stride(addmm_44, (512, 4096), (4096, 1))
    assert_size_stride(getitem_275, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_872, (512, 4096), (4096, 1))
    assert_size_stride(getitem_277, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_186, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_281, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(getitem_283, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_189, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_908, (512, 1024), (1024, 1))
    assert_size_stride(addmm_46, (512, 4096), (4096, 1))
    assert_size_stride(getitem_287, (512, 1, 4096), (4096, 4096, 1))
    assert_size_stride(view_910, (512, 4096), (4096, 1))
    assert_size_stride(getitem_289, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(mul_194, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(getitem_293, (512, 1, 1024), (1024, 1024, 1))
    assert_size_stride(view_912, (512, 1024), (1024, 1))
    assert_size_stride(sub_73, (512, 32000), (32000, 1))
    assert_size_stride(convert_element_type_5, (), ())
    assert_size_stride(permute_1013, (32000, 1024), (1024, 1))
    assert_size_stride(div_27, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1018, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1022, (4096, 1024), (1024, 1))
    assert_size_stride(div_28, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1027, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1028, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1034, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1035, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_26, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1041, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1042, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1048, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1049, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1055, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1059, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1060, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1067, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1074, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_29, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1079, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1083, (4096, 1024), (1024, 1))
    assert_size_stride(div_30, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1088, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1089, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1095, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1096, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_27, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1102, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1103, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1109, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1110, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1120, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1121, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1128, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1135, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_31, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1140, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1144, (4096, 1024), (1024, 1))
    assert_size_stride(div_32, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1149, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1150, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1156, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1157, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_28, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1163, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1164, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1170, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1171, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1181, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1182, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1189, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1196, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_33, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1201, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1205, (4096, 1024), (1024, 1))
    assert_size_stride(div_34, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1210, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1211, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1217, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1218, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_29, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1224, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1225, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1231, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1232, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1242, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1243, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1250, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1257, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_35, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1262, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1266, (4096, 1024), (1024, 1))
    assert_size_stride(div_36, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1271, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1272, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1278, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1279, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_30, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1285, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1286, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1292, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1293, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1303, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1304, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1311, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1318, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_37, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1323, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1327, (4096, 1024), (1024, 1))
    assert_size_stride(div_38, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1332, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1333, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1339, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1340, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_31, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1346, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1347, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1353, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1354, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1364, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1365, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1372, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1379, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_39, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1384, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1388, (4096, 1024), (1024, 1))
    assert_size_stride(div_40, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1393, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1394, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1400, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1401, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_32, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1407, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1408, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1414, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1415, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1425, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1426, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1433, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1440, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_41, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1445, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1449, (4096, 1024), (1024, 1))
    assert_size_stride(div_42, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1454, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1455, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1461, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1462, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_33, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1468, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1469, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1475, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1476, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1486, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1487, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1494, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1501, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_43, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1506, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1510, (4096, 1024), (1024, 1))
    assert_size_stride(div_44, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1515, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1516, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1522, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1523, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_34, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1529, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1530, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1536, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1537, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1547, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1548, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1555, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1562, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_45, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1567, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1571, (4096, 1024), (1024, 1))
    assert_size_stride(div_46, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1576, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1577, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1583, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1584, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_35, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1590, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1591, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1597, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1598, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1608, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1609, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1616, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1623, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_47, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1628, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1632, (4096, 1024), (1024, 1))
    assert_size_stride(div_48, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1637, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1638, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1644, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1645, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_36, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1651, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1652, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1658, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1659, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1669, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1670, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1677, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1684, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_49, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1689, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1693, (4096, 1024), (1024, 1))
    assert_size_stride(div_50, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1698, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1699, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1705, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1706, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_37, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1712, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1713, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1719, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1720, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1730, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1731, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1738, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1745, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_51, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1750, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1754, (4096, 1024), (1024, 1))
    assert_size_stride(div_52, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1759, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1760, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1766, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1767, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_38, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1773, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1774, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1780, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1781, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1791, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1792, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1799, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1806, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_53, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1811, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1815, (4096, 1024), (1024, 1))
    assert_size_stride(div_54, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1820, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1821, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1827, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1828, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_39, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1834, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1835, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1841, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1842, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1852, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1853, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1860, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1867, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_55, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1872, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1876, (4096, 1024), (1024, 1))
    assert_size_stride(div_56, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1881, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1882, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1888, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1889, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_40, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1895, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1896, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1902, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1903, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1913, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1914, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1921, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1928, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_57, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1933, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1937, (4096, 1024), (1024, 1))
    assert_size_stride(div_58, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1942, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1943, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_1949, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_1950, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_41, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_1956, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1957, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_1963, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_1964, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_1974, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_1975, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1982, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_1989, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_59, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_1994, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1998, (4096, 1024), (1024, 1))
    assert_size_stride(div_60, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2003, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2004, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2010, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2011, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_42, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2017, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2018, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2024, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2025, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2035, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2036, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2043, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2050, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_61, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2055, (1024, 4096), (4096, 1))
    assert_size_stride(permute_2059, (4096, 1024), (1024, 1))
    assert_size_stride(div_62, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2064, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2065, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2071, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2072, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_43, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2078, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2079, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2085, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2086, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2096, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2097, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2104, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2111, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_63, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2116, (1024, 4096), (4096, 1))
    assert_size_stride(permute_2120, (4096, 1024), (1024, 1))
    assert_size_stride(div_64, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2125, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2126, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2132, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2133, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_44, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2139, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2140, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2146, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2147, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2157, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2158, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2165, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2172, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_65, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2177, (1024, 4096), (4096, 1))
    assert_size_stride(permute_2181, (4096, 1024), (1024, 1))
    assert_size_stride(div_66, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2186, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2187, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2193, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2194, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_45, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2200, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2201, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2207, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2208, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2218, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2219, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2226, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2233, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_67, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2238, (1024, 4096), (4096, 1))
    assert_size_stride(permute_2242, (4096, 1024), (1024, 1))
    assert_size_stride(div_68, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2247, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2248, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2254, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2255, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_46, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2261, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2262, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2268, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2269, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2279, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2280, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2287, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2294, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_69, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2299, (1024, 4096), (4096, 1))
    assert_size_stride(permute_2303, (4096, 1024), (1024, 1))
    assert_size_stride(div_70, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2308, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2309, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2315, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2316, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_47, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2322, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2323, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2329, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2330, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2340, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2341, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2348, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2355, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_71, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2360, (1024, 4096), (4096, 1))
    assert_size_stride(permute_2364, (4096, 1024), (1024, 1))
    assert_size_stride(div_72, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2369, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2370, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2376, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2377, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_48, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2383, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2384, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2390, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2391, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2401, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2402, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2409, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2416, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(div_73, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2421, (1024, 4096), (4096, 1))
    assert_size_stride(permute_2425, (4096, 1024), (1024, 1))
    assert_size_stride(div_74, (512, 1, 1), (1, 1, 1))
    assert_size_stride(permute_2430, (1, 1024, 512), (0, 1, 1024))
    assert_size_stride(permute_2431, (1, 1024, 1024), (0, 1, 1024))
    assert_size_stride(permute_2437, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_2438, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(alias_49, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_2444, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2445, (16, 1024, 64), (64, 1024, 1))
    assert_size_stride(permute_2451, (16, 64, 512), (64, 1, 1024))
    assert_size_stride(permute_2452, (16, 512, 64), (64, 1024, 1))
    assert_size_stride(permute_2462, (1, 1024, 512), (524288, 1, 1024))
    assert_size_stride(permute_2463, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2470, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_2477, (1, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 32000), (16384000, 32000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 16384000, grid=grid(16384000), stream=stream0)
        buf1 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_365, buf1, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_365, tangents_1, convert_element_type_5, tangents_2, sub_73, buf5, 512, 32000, grid=grid(512), stream=stream0)
        del buf0
        del convert_element_type_5
        del primals_365
        del sub_73
        del tangents_1
        del tangents_2
        buf6 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 32000), (32000, 1), 0), permute_1013, out=buf6)
        del permute_1013
        buf7 = empty((32000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (32000, 512), (1, 32000), 0), view_912, out=buf7)
        del view_912
        buf8 = empty((1, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf5, buf8, 32000, 512, grid=grid(32000), stream=stream0)
        del buf5
        buf11 = empty_strided((512, 1, 1024), (1024, 524288, 1), device='cuda', dtype=torch.float32)
        buf14 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_4.run(buf6, getitem_293, primals_360, mul_194, div_27, getitem_289, buf11, buf14, 512, 1024, grid=grid(512), stream=stream0)
        del div_27
        del getitem_289
        del primals_360
        buf12 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf13 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_5.run(buf6, getitem_293, mul_194, buf12, buf13, 1024, 512, grid=grid(1024), stream=stream0)
        del getitem_293
        del mul_194
        buf15 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (512, 1024), (1024, 1), 0), permute_1018, out=buf15)
        del permute_1018
        buf16 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (1024, 512), (1, 1024), 0), view_910, out=buf16)
        del view_910
        buf17 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf14, buf17, 4096, 128, grid=grid(4096), stream=stream0)
        buf18 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf17, buf18, 1024, 4, grid=grid(1024), stream=stream0)
        buf19 = reinterpret_tensor(buf15, (512, 1, 4096), (4096, 4096, 1), 0); del buf15  # reuse
        # Source Nodes: [output_187], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf19, getitem_287, addmm_46, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_46
        del getitem_287
        buf20 = reinterpret_tensor(buf14, (512, 1024), (1024, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (512, 4096), (4096, 1), 0), permute_1022, out=buf20)
        del permute_1022
        buf21 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (4096, 512), (1, 4096), 0), view_908, out=buf21)
        del view_908
        buf22 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf19, buf22, 16384, 128, grid=grid(16384), stream=stream0)
        buf23 = reinterpret_tensor(buf17, (1, 4096), (4096, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf22, buf23, 4096, 4, grid=grid(4096), stream=stream0)
        buf26 = reinterpret_tensor(buf6, (512, 1, 1024), (1024, 524288, 1), 0); del buf6  # reuse
        buf29 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf11, buf20, primals_354, mul_189, div_28, getitem_283, buf26, buf29, 512, 1024, grid=grid(512), stream=stream0)
        del div_28
        del getitem_283
        del primals_354
        buf27 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf28 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf11, buf20, mul_189, buf27, buf28, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_189
        buf30 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1027, reinterpret_tensor(buf29, (1, 512, 1024), (0, 1024, 1), 0), out=buf30)
        del permute_1027
        buf31 = reinterpret_tensor(buf20, (1, 512, 1024), (524288, 1024, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (1, 512, 1024), (0, 1024, 1), 0), permute_1028, out=buf31)
        del permute_1028
        buf32 = reinterpret_tensor(buf29, (16, 512, 64), (32768, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1034, reinterpret_tensor(buf31, (16, 512, 64), (1, 1024, 16), 0), out=buf32)
        del permute_1034
        buf33 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (16, 512, 64), (1, 1024, 16), 0), permute_1035, out=buf33)
        del permute_1035
        buf36 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf33, getitem_281, alias_26, buf36, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_26
        del getitem_281
        buf35 = empty((1, 16, 512, 1023), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf35, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf35, [None, None, None, iota_2], buf36, True)
        buf40 = empty((16, 512, 1024), device='cuda', dtype=torch.float32)
        buf42 = empty((16, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf35, buf40, buf42, 8388608, grid=grid(8388608), stream=stream0)
        buf41 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1041, buf40, out=buf41)
        del permute_1041
        buf43 = reinterpret_tensor(buf31, (16, 512, 64), (32768, 64, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf42, permute_1042, out=buf43)
        del permute_1042
        buf44 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf43, buf44, 1024, 512, grid=grid(1024), stream=stream0)
        buf45 = reinterpret_tensor(buf11, (16, 64, 512), (32768, 512, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1048, reinterpret_tensor(buf36, (16, 512, 512), (262144, 512, 1), 0), out=buf45)
        del permute_1048
        buf46 = empty((16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (16, 512, 512), (262144, 512, 1), 0), permute_1049, out=buf46)
        del permute_1049
        buf47 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf46, buf47, 1024, 512, grid=grid(1024), stream=stream0)
        buf48 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf41, (1, 1024, 1024), (0, 1, 1024), 0), out=buf48)
        buf49 = empty((512, 1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf32, buf49, 524288, grid=grid(524288), stream=stream0)
        buf50 = reinterpret_tensor(buf41, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1059, reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), out=buf50)
        buf51 = reinterpret_tensor(buf32, (1, 512, 1024), (524288, 1024, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), permute_1060, out=buf51)
        del permute_1060
        buf52 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1059, reinterpret_tensor(buf45, (1, 512, 1024), (0, 1, 512), 0), out=buf52)
        buf53 = reinterpret_tensor(buf49, (1, 512, 1024), (524288, 1024, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf45, (1, 512, 1024), (0, 1, 512), 0), permute_1067, out=buf53)
        del permute_1067
        buf54 = reinterpret_tensor(buf45, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf43, buf46, buf54, 524288, grid=grid(524288), stream=stream0)
        buf55 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1059, reinterpret_tensor(buf54, (1, 512, 1024), (0, 1024, 1), 0), out=buf55)
        del permute_1059
        buf56 = reinterpret_tensor(buf46, (1, 512, 1024), (524288, 1024, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (1, 512, 1024), (0, 1024, 1), 0), permute_1074, out=buf56)
        del permute_1074
        buf60 = reinterpret_tensor(buf54, (512, 1, 1024), (1024, 524288, 1), 0); del buf54  # reuse
        buf63 = reinterpret_tensor(buf43, (512, 1, 1024), (1024, 1024, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf26, buf51, buf53, buf56, primals_352, mul_186, div_29, getitem_277, buf60, buf63, 512, 1024, grid=grid(512), stream=stream0)
        del div_29
        del getitem_277
        del primals_352
        buf61 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf62 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf26, buf51, buf53, buf56, mul_186, buf61, buf62, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_186
        buf64 = reinterpret_tensor(buf19, (512, 4096), (4096, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (512, 1024), (1024, 1), 0), permute_1079, out=buf64)
        del permute_1079
        buf65 = reinterpret_tensor(buf36, (1024, 4096), (4096, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (1024, 512), (1, 1024), 0), view_872, out=buf65)
        del view_872
        buf66 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf63, buf66, 4096, 128, grid=grid(4096), stream=stream0)
        buf67 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf66, buf67, 1024, 4, grid=grid(1024), stream=stream0)
        buf68 = reinterpret_tensor(buf64, (512, 1, 4096), (4096, 4096, 1), 0); del buf64  # reuse
        # Source Nodes: [output_179], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf68, getitem_275, addmm_44, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_44
        del getitem_275
        buf69 = reinterpret_tensor(buf63, (512, 1024), (1024, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (512, 4096), (4096, 1), 0), permute_1083, out=buf69)
        del permute_1083
        buf70 = reinterpret_tensor(buf33, (4096, 1024), (1024, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (4096, 512), (1, 4096), 0), view_870, out=buf70)
        del view_870
        buf71 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf68, buf71, 16384, 128, grid=grid(16384), stream=stream0)
        buf72 = reinterpret_tensor(buf66, (1, 4096), (4096, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf71, buf72, 4096, 4, grid=grid(4096), stream=stream0)
        buf75 = reinterpret_tensor(buf56, (512, 1, 1024), (1024, 524288, 1), 0); del buf56  # reuse
        buf78 = reinterpret_tensor(buf53, (512, 1, 1024), (1024, 1024, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf60, buf69, primals_346, mul_181, div_30, getitem_271, buf75, buf78, 512, 1024, grid=grid(512), stream=stream0)
        del div_30
        del getitem_271
        del primals_346
        buf76 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf77 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf60, buf69, mul_181, buf76, buf77, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_181
        buf79 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1088, reinterpret_tensor(buf78, (1, 512, 1024), (0, 1024, 1), 0), out=buf79)
        del permute_1088
        buf80 = reinterpret_tensor(buf69, (1, 512, 1024), (524288, 1024, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (1, 512, 1024), (0, 1024, 1), 0), permute_1089, out=buf80)
        del permute_1089
        buf81 = reinterpret_tensor(buf78, (16, 512, 64), (32768, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1095, reinterpret_tensor(buf80, (16, 512, 64), (1, 1024, 16), 0), out=buf81)
        del permute_1095
        buf82 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (16, 512, 64), (1, 1024, 16), 0), permute_1096, out=buf82)
        del permute_1096
        buf85 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf82, getitem_269, alias_27, buf85, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_27
        del getitem_269
        buf84 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf84, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf84, [None, None, None, iota_2], buf85, True)
        buf89 = buf42; del buf42  # reuse
        buf91 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf84, buf89, buf91, 8388608, grid=grid(8388608), stream=stream0)
        buf90 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1102, buf89, out=buf90)
        del permute_1102
        buf92 = reinterpret_tensor(buf80, (16, 512, 64), (32768, 64, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf91, permute_1103, out=buf92)
        del permute_1103
        buf93 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf92, buf93, 1024, 512, grid=grid(1024), stream=stream0)
        buf94 = reinterpret_tensor(buf60, (16, 64, 512), (32768, 512, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1109, reinterpret_tensor(buf85, (16, 512, 512), (262144, 512, 1), 0), out=buf94)
        del permute_1109
        buf95 = reinterpret_tensor(buf51, (16, 512, 64), (32768, 64, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (16, 512, 512), (262144, 512, 1), 0), permute_1110, out=buf95)
        del permute_1110
        buf96 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf95, buf96, 1024, 512, grid=grid(1024), stream=stream0)
        buf97 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf90, (1, 1024, 1024), (0, 1, 1024), 0), out=buf97)
        buf98 = reinterpret_tensor(buf26, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf81, buf98, 524288, grid=grid(524288), stream=stream0)
        buf99 = reinterpret_tensor(buf90, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1120, reinterpret_tensor(buf98, (1, 512, 1024), (0, 1024, 1), 0), out=buf99)
        buf100 = reinterpret_tensor(buf81, (1, 512, 1024), (524288, 1024, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (1, 512, 1024), (0, 1024, 1), 0), permute_1121, out=buf100)
        del permute_1121
        buf101 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1120, reinterpret_tensor(buf94, (1, 512, 1024), (0, 1, 512), 0), out=buf101)
        buf102 = reinterpret_tensor(buf98, (1, 512, 1024), (524288, 1024, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (1, 512, 1024), (0, 1, 512), 0), permute_1128, out=buf102)
        del permute_1128
        buf103 = reinterpret_tensor(buf94, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf92, buf95, buf103, 524288, grid=grid(524288), stream=stream0)
        buf104 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1120, reinterpret_tensor(buf103, (1, 512, 1024), (0, 1024, 1), 0), out=buf104)
        del permute_1120
        buf105 = reinterpret_tensor(buf95, (1, 512, 1024), (524288, 1024, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf103, (1, 512, 1024), (0, 1024, 1), 0), permute_1135, out=buf105)
        del permute_1135
        buf109 = reinterpret_tensor(buf103, (512, 1, 1024), (1024, 524288, 1), 0); del buf103  # reuse
        buf112 = reinterpret_tensor(buf92, (512, 1, 1024), (1024, 1024, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf75, buf100, buf102, buf105, primals_344, mul_178, div_31, getitem_265, buf109, buf112, 512, 1024, grid=grid(512), stream=stream0)
        del div_31
        del getitem_265
        del primals_344
        buf110 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf111 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf75, buf100, buf102, buf105, mul_178, buf110, buf111, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_178
        buf113 = reinterpret_tensor(buf68, (512, 4096), (4096, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (512, 1024), (1024, 1), 0), permute_1140, out=buf113)
        del permute_1140
        buf114 = reinterpret_tensor(buf85, (1024, 4096), (4096, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (1024, 512), (1, 1024), 0), view_834, out=buf114)
        del view_834
        buf115 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf112, buf115, 4096, 128, grid=grid(4096), stream=stream0)
        buf116 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf115, buf116, 1024, 4, grid=grid(1024), stream=stream0)
        buf117 = reinterpret_tensor(buf113, (512, 1, 4096), (4096, 4096, 1), 0); del buf113  # reuse
        # Source Nodes: [output_171], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf117, getitem_263, addmm_42, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_42
        del getitem_263
        buf118 = reinterpret_tensor(buf112, (512, 1024), (1024, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (512, 4096), (4096, 1), 0), permute_1144, out=buf118)
        del permute_1144
        buf119 = reinterpret_tensor(buf82, (4096, 1024), (1024, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (4096, 512), (1, 4096), 0), view_832, out=buf119)
        del view_832
        buf120 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf117, buf120, 16384, 128, grid=grid(16384), stream=stream0)
        buf121 = reinterpret_tensor(buf115, (1, 4096), (4096, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf120, buf121, 4096, 4, grid=grid(4096), stream=stream0)
        buf124 = buf75; del buf75  # reuse
        buf127 = reinterpret_tensor(buf105, (512, 1, 1024), (1024, 1024, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf109, buf118, primals_338, mul_173, div_32, getitem_259, buf124, buf127, 512, 1024, grid=grid(512), stream=stream0)
        del div_32
        del getitem_259
        del primals_338
        buf125 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf126 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf109, buf118, mul_173, buf125, buf126, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_173
        buf128 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1149, reinterpret_tensor(buf127, (1, 512, 1024), (0, 1024, 1), 0), out=buf128)
        del permute_1149
        buf129 = reinterpret_tensor(buf118, (1, 512, 1024), (524288, 1024, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf127, (1, 512, 1024), (0, 1024, 1), 0), permute_1150, out=buf129)
        del permute_1150
        buf130 = reinterpret_tensor(buf127, (16, 512, 64), (32768, 64, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1156, reinterpret_tensor(buf129, (16, 512, 64), (1, 1024, 16), 0), out=buf130)
        del permute_1156
        buf131 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (16, 512, 64), (1, 1024, 16), 0), permute_1157, out=buf131)
        del permute_1157
        buf134 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf131, getitem_257, alias_28, buf134, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_28
        del getitem_257
        buf133 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf133, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf133, [None, None, None, iota_2], buf134, True)
        buf138 = buf91; del buf91  # reuse
        buf140 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf133, buf138, buf140, 8388608, grid=grid(8388608), stream=stream0)
        buf139 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1163, buf138, out=buf139)
        del permute_1163
        buf141 = reinterpret_tensor(buf129, (16, 512, 64), (32768, 64, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf140, permute_1164, out=buf141)
        del permute_1164
        buf142 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf141, buf142, 1024, 512, grid=grid(1024), stream=stream0)
        buf143 = reinterpret_tensor(buf109, (16, 64, 512), (32768, 512, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1170, reinterpret_tensor(buf134, (16, 512, 512), (262144, 512, 1), 0), out=buf143)
        del permute_1170
        buf144 = reinterpret_tensor(buf102, (16, 512, 64), (32768, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf134, (16, 512, 512), (262144, 512, 1), 0), permute_1171, out=buf144)
        del permute_1171
        buf145 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf144, buf145, 1024, 512, grid=grid(1024), stream=stream0)
        buf146 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf139, (1, 1024, 1024), (0, 1, 1024), 0), out=buf146)
        buf147 = reinterpret_tensor(buf100, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf130, buf147, 524288, grid=grid(524288), stream=stream0)
        buf148 = reinterpret_tensor(buf139, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1181, reinterpret_tensor(buf147, (1, 512, 1024), (0, 1024, 1), 0), out=buf148)
        buf149 = reinterpret_tensor(buf130, (1, 512, 1024), (524288, 1024, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (1, 512, 1024), (0, 1024, 1), 0), permute_1182, out=buf149)
        del permute_1182
        buf150 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1181, reinterpret_tensor(buf143, (1, 512, 1024), (0, 1, 512), 0), out=buf150)
        buf151 = reinterpret_tensor(buf147, (1, 512, 1024), (524288, 1024, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (1, 512, 1024), (0, 1, 512), 0), permute_1189, out=buf151)
        del permute_1189
        buf152 = reinterpret_tensor(buf143, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf141, buf144, buf152, 524288, grid=grid(524288), stream=stream0)
        buf153 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1181, reinterpret_tensor(buf152, (1, 512, 1024), (0, 1024, 1), 0), out=buf153)
        del permute_1181
        buf154 = reinterpret_tensor(buf144, (1, 512, 1024), (524288, 1024, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf152, (1, 512, 1024), (0, 1024, 1), 0), permute_1196, out=buf154)
        del permute_1196
        buf158 = reinterpret_tensor(buf152, (512, 1, 1024), (1024, 524288, 1), 0); del buf152  # reuse
        buf161 = reinterpret_tensor(buf141, (512, 1, 1024), (1024, 1024, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf124, buf149, buf151, buf154, primals_336, mul_170, div_33, getitem_253, buf158, buf161, 512, 1024, grid=grid(512), stream=stream0)
        del div_33
        del getitem_253
        del primals_336
        buf159 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf160 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf124, buf149, buf151, buf154, mul_170, buf159, buf160, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_170
        buf162 = reinterpret_tensor(buf117, (512, 4096), (4096, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 1024), (1024, 1), 0), permute_1201, out=buf162)
        del permute_1201
        buf163 = reinterpret_tensor(buf134, (1024, 4096), (4096, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (1024, 512), (1, 1024), 0), view_796, out=buf163)
        del view_796
        buf164 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf161, buf164, 4096, 128, grid=grid(4096), stream=stream0)
        buf165 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf164, buf165, 1024, 4, grid=grid(1024), stream=stream0)
        buf166 = reinterpret_tensor(buf162, (512, 1, 4096), (4096, 4096, 1), 0); del buf162  # reuse
        # Source Nodes: [output_163], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf166, getitem_251, addmm_40, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_40
        del getitem_251
        buf167 = reinterpret_tensor(buf161, (512, 1024), (1024, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0), permute_1205, out=buf167)
        del permute_1205
        buf168 = reinterpret_tensor(buf131, (4096, 1024), (1024, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (4096, 512), (1, 4096), 0), view_794, out=buf168)
        del view_794
        buf169 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf166, buf169, 16384, 128, grid=grid(16384), stream=stream0)
        buf170 = reinterpret_tensor(buf164, (1, 4096), (4096, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf169, buf170, 4096, 4, grid=grid(4096), stream=stream0)
        buf173 = reinterpret_tensor(buf154, (512, 1, 1024), (1024, 524288, 1), 0); del buf154  # reuse
        buf176 = reinterpret_tensor(buf151, (512, 1, 1024), (1024, 1024, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf158, buf167, primals_330, mul_165, div_34, getitem_247, buf173, buf176, 512, 1024, grid=grid(512), stream=stream0)
        del div_34
        del getitem_247
        del primals_330
        buf174 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf175 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf158, buf167, mul_165, buf174, buf175, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_165
        buf177 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1210, reinterpret_tensor(buf176, (1, 512, 1024), (0, 1024, 1), 0), out=buf177)
        del permute_1210
        buf178 = reinterpret_tensor(buf167, (1, 512, 1024), (524288, 1024, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (1, 512, 1024), (0, 1024, 1), 0), permute_1211, out=buf178)
        del permute_1211
        buf179 = reinterpret_tensor(buf176, (16, 512, 64), (32768, 64, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1217, reinterpret_tensor(buf178, (16, 512, 64), (1, 1024, 16), 0), out=buf179)
        del permute_1217
        buf180 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf178, (16, 512, 64), (1, 1024, 16), 0), permute_1218, out=buf180)
        del permute_1218
        buf183 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf180, getitem_245, alias_29, buf183, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_29
        del getitem_245
        buf182 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf182, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf182, [None, None, None, iota_2], buf183, True)
        buf187 = buf140; del buf140  # reuse
        buf189 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf182, buf187, buf189, 8388608, grid=grid(8388608), stream=stream0)
        buf188 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1224, buf187, out=buf188)
        del permute_1224
        buf190 = reinterpret_tensor(buf178, (16, 512, 64), (32768, 64, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf189, permute_1225, out=buf190)
        del permute_1225
        buf191 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf190, buf191, 1024, 512, grid=grid(1024), stream=stream0)
        buf192 = reinterpret_tensor(buf158, (16, 64, 512), (32768, 512, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1231, reinterpret_tensor(buf183, (16, 512, 512), (262144, 512, 1), 0), out=buf192)
        del permute_1231
        buf193 = reinterpret_tensor(buf149, (16, 512, 64), (32768, 64, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (16, 512, 512), (262144, 512, 1), 0), permute_1232, out=buf193)
        del permute_1232
        buf194 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf193, buf194, 1024, 512, grid=grid(1024), stream=stream0)
        buf195 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf188, (1, 1024, 1024), (0, 1, 1024), 0), out=buf195)
        buf196 = reinterpret_tensor(buf124, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf179, buf196, 524288, grid=grid(524288), stream=stream0)
        buf197 = reinterpret_tensor(buf188, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1242, reinterpret_tensor(buf196, (1, 512, 1024), (0, 1024, 1), 0), out=buf197)
        buf198 = reinterpret_tensor(buf179, (1, 512, 1024), (524288, 1024, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf196, (1, 512, 1024), (0, 1024, 1), 0), permute_1243, out=buf198)
        del permute_1243
        buf199 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1242, reinterpret_tensor(buf192, (1, 512, 1024), (0, 1, 512), 0), out=buf199)
        buf200 = reinterpret_tensor(buf196, (1, 512, 1024), (524288, 1024, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf192, (1, 512, 1024), (0, 1, 512), 0), permute_1250, out=buf200)
        del permute_1250
        buf201 = reinterpret_tensor(buf192, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf190, buf193, buf201, 524288, grid=grid(524288), stream=stream0)
        buf202 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1242, reinterpret_tensor(buf201, (1, 512, 1024), (0, 1024, 1), 0), out=buf202)
        del permute_1242
        buf203 = reinterpret_tensor(buf193, (1, 512, 1024), (524288, 1024, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (1, 512, 1024), (0, 1024, 1), 0), permute_1257, out=buf203)
        del permute_1257
        buf207 = reinterpret_tensor(buf201, (512, 1, 1024), (1024, 524288, 1), 0); del buf201  # reuse
        buf210 = reinterpret_tensor(buf190, (512, 1, 1024), (1024, 1024, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf173, buf198, buf200, buf203, primals_328, mul_162, div_35, getitem_241, buf207, buf210, 512, 1024, grid=grid(512), stream=stream0)
        del div_35
        del getitem_241
        del primals_328
        buf208 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf209 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf173, buf198, buf200, buf203, mul_162, buf208, buf209, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_162
        buf211 = reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (512, 1024), (1024, 1), 0), permute_1262, out=buf211)
        del permute_1262
        buf212 = reinterpret_tensor(buf183, (1024, 4096), (4096, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (1024, 512), (1, 1024), 0), view_758, out=buf212)
        del view_758
        buf213 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf210, buf213, 4096, 128, grid=grid(4096), stream=stream0)
        buf214 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf213, buf214, 1024, 4, grid=grid(1024), stream=stream0)
        buf215 = reinterpret_tensor(buf211, (512, 1, 4096), (4096, 4096, 1), 0); del buf211  # reuse
        # Source Nodes: [output_155], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf215, getitem_239, addmm_38, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_38
        del getitem_239
        buf216 = reinterpret_tensor(buf210, (512, 1024), (1024, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (512, 4096), (4096, 1), 0), permute_1266, out=buf216)
        del permute_1266
        buf217 = reinterpret_tensor(buf180, (4096, 1024), (1024, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (4096, 512), (1, 4096), 0), view_756, out=buf217)
        del view_756
        buf218 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf215, buf218, 16384, 128, grid=grid(16384), stream=stream0)
        buf219 = reinterpret_tensor(buf213, (1, 4096), (4096, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf218, buf219, 4096, 4, grid=grid(4096), stream=stream0)
        buf222 = reinterpret_tensor(buf203, (512, 1, 1024), (1024, 524288, 1), 0); del buf203  # reuse
        buf225 = reinterpret_tensor(buf200, (512, 1, 1024), (1024, 1024, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf207, buf216, primals_322, mul_157, div_36, getitem_235, buf222, buf225, 512, 1024, grid=grid(512), stream=stream0)
        del div_36
        del getitem_235
        del primals_322
        buf223 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf224 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf207, buf216, mul_157, buf223, buf224, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_157
        buf226 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1271, reinterpret_tensor(buf225, (1, 512, 1024), (0, 1024, 1), 0), out=buf226)
        del permute_1271
        buf227 = reinterpret_tensor(buf216, (1, 512, 1024), (524288, 1024, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (1, 512, 1024), (0, 1024, 1), 0), permute_1272, out=buf227)
        del permute_1272
        buf228 = reinterpret_tensor(buf225, (16, 512, 64), (32768, 64, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1278, reinterpret_tensor(buf227, (16, 512, 64), (1, 1024, 16), 0), out=buf228)
        del permute_1278
        buf229 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf227, (16, 512, 64), (1, 1024, 16), 0), permute_1279, out=buf229)
        del permute_1279
        buf232 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf229, getitem_233, alias_30, buf232, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_30
        del getitem_233
        buf231 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf231, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf231, [None, None, None, iota_2], buf232, True)
        buf236 = buf189; del buf189  # reuse
        buf238 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf231, buf236, buf238, 8388608, grid=grid(8388608), stream=stream0)
        buf237 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1285, buf236, out=buf237)
        del permute_1285
        buf239 = reinterpret_tensor(buf227, (16, 512, 64), (32768, 64, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf238, permute_1286, out=buf239)
        del permute_1286
        buf240 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf239, buf240, 1024, 512, grid=grid(1024), stream=stream0)
        buf241 = reinterpret_tensor(buf207, (16, 64, 512), (32768, 512, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1292, reinterpret_tensor(buf232, (16, 512, 512), (262144, 512, 1), 0), out=buf241)
        del permute_1292
        buf242 = reinterpret_tensor(buf198, (16, 512, 64), (32768, 64, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (16, 512, 512), (262144, 512, 1), 0), permute_1293, out=buf242)
        del permute_1293
        buf243 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf242, buf243, 1024, 512, grid=grid(1024), stream=stream0)
        buf244 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf237, (1, 1024, 1024), (0, 1, 1024), 0), out=buf244)
        buf245 = reinterpret_tensor(buf173, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf228, buf245, 524288, grid=grid(524288), stream=stream0)
        buf246 = reinterpret_tensor(buf237, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1303, reinterpret_tensor(buf245, (1, 512, 1024), (0, 1024, 1), 0), out=buf246)
        buf247 = reinterpret_tensor(buf228, (1, 512, 1024), (524288, 1024, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf245, (1, 512, 1024), (0, 1024, 1), 0), permute_1304, out=buf247)
        del permute_1304
        buf248 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1303, reinterpret_tensor(buf241, (1, 512, 1024), (0, 1, 512), 0), out=buf248)
        buf249 = reinterpret_tensor(buf245, (1, 512, 1024), (524288, 1024, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (1, 512, 1024), (0, 1, 512), 0), permute_1311, out=buf249)
        del permute_1311
        buf250 = reinterpret_tensor(buf241, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf239, buf242, buf250, 524288, grid=grid(524288), stream=stream0)
        buf251 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1303, reinterpret_tensor(buf250, (1, 512, 1024), (0, 1024, 1), 0), out=buf251)
        del permute_1303
        buf252 = reinterpret_tensor(buf242, (1, 512, 1024), (524288, 1024, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf250, (1, 512, 1024), (0, 1024, 1), 0), permute_1318, out=buf252)
        del permute_1318
        buf256 = reinterpret_tensor(buf250, (512, 1, 1024), (1024, 524288, 1), 0); del buf250  # reuse
        buf259 = reinterpret_tensor(buf239, (512, 1, 1024), (1024, 1024, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf222, buf247, buf249, buf252, primals_320, mul_154, div_37, getitem_229, buf256, buf259, 512, 1024, grid=grid(512), stream=stream0)
        del div_37
        del getitem_229
        del primals_320
        buf257 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf258 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf222, buf247, buf249, buf252, mul_154, buf257, buf258, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_154
        buf260 = reinterpret_tensor(buf215, (512, 4096), (4096, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (512, 1024), (1024, 1), 0), permute_1323, out=buf260)
        del permute_1323
        buf261 = reinterpret_tensor(buf232, (1024, 4096), (4096, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (1024, 512), (1, 1024), 0), view_720, out=buf261)
        del view_720
        buf262 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf259, buf262, 4096, 128, grid=grid(4096), stream=stream0)
        buf263 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf262, buf263, 1024, 4, grid=grid(1024), stream=stream0)
        buf264 = reinterpret_tensor(buf260, (512, 1, 4096), (4096, 4096, 1), 0); del buf260  # reuse
        # Source Nodes: [output_147], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf264, getitem_227, addmm_36, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_36
        del getitem_227
        buf265 = reinterpret_tensor(buf259, (512, 1024), (1024, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (512, 4096), (4096, 1), 0), permute_1327, out=buf265)
        del permute_1327
        buf266 = reinterpret_tensor(buf229, (4096, 1024), (1024, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (4096, 512), (1, 4096), 0), view_718, out=buf266)
        del view_718
        buf267 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf264, buf267, 16384, 128, grid=grid(16384), stream=stream0)
        buf268 = reinterpret_tensor(buf262, (1, 4096), (4096, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf267, buf268, 4096, 4, grid=grid(4096), stream=stream0)
        buf271 = reinterpret_tensor(buf252, (512, 1, 1024), (1024, 524288, 1), 0); del buf252  # reuse
        buf274 = reinterpret_tensor(buf249, (512, 1, 1024), (1024, 1024, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf256, buf265, primals_314, mul_149, div_38, getitem_223, buf271, buf274, 512, 1024, grid=grid(512), stream=stream0)
        del div_38
        del getitem_223
        del primals_314
        buf272 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf273 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf256, buf265, mul_149, buf272, buf273, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_149
        buf275 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1332, reinterpret_tensor(buf274, (1, 512, 1024), (0, 1024, 1), 0), out=buf275)
        del permute_1332
        buf276 = reinterpret_tensor(buf265, (1, 512, 1024), (524288, 1024, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf274, (1, 512, 1024), (0, 1024, 1), 0), permute_1333, out=buf276)
        del permute_1333
        buf277 = reinterpret_tensor(buf274, (16, 512, 64), (32768, 64, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1339, reinterpret_tensor(buf276, (16, 512, 64), (1, 1024, 16), 0), out=buf277)
        del permute_1339
        buf278 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (16, 512, 64), (1, 1024, 16), 0), permute_1340, out=buf278)
        del permute_1340
        buf281 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf278, getitem_221, alias_31, buf281, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_31
        del getitem_221
        buf280 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf280, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf280, [None, None, None, iota_2], buf281, True)
        buf285 = buf238; del buf238  # reuse
        buf287 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf280, buf285, buf287, 8388608, grid=grid(8388608), stream=stream0)
        buf286 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1346, buf285, out=buf286)
        del permute_1346
        buf288 = reinterpret_tensor(buf276, (16, 512, 64), (32768, 64, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf287, permute_1347, out=buf288)
        del permute_1347
        buf289 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf288, buf289, 1024, 512, grid=grid(1024), stream=stream0)
        buf290 = reinterpret_tensor(buf256, (16, 64, 512), (32768, 512, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1353, reinterpret_tensor(buf281, (16, 512, 512), (262144, 512, 1), 0), out=buf290)
        del permute_1353
        buf291 = reinterpret_tensor(buf247, (16, 512, 64), (32768, 64, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf281, (16, 512, 512), (262144, 512, 1), 0), permute_1354, out=buf291)
        del permute_1354
        buf292 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf291, buf292, 1024, 512, grid=grid(1024), stream=stream0)
        buf293 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf286, (1, 1024, 1024), (0, 1, 1024), 0), out=buf293)
        buf294 = reinterpret_tensor(buf222, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf277, buf294, 524288, grid=grid(524288), stream=stream0)
        buf295 = reinterpret_tensor(buf286, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1364, reinterpret_tensor(buf294, (1, 512, 1024), (0, 1024, 1), 0), out=buf295)
        buf296 = reinterpret_tensor(buf277, (1, 512, 1024), (524288, 1024, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (1, 512, 1024), (0, 1024, 1), 0), permute_1365, out=buf296)
        del permute_1365
        buf297 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1364, reinterpret_tensor(buf290, (1, 512, 1024), (0, 1, 512), 0), out=buf297)
        buf298 = reinterpret_tensor(buf294, (1, 512, 1024), (524288, 1024, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf290, (1, 512, 1024), (0, 1, 512), 0), permute_1372, out=buf298)
        del permute_1372
        buf299 = reinterpret_tensor(buf290, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf288, buf291, buf299, 524288, grid=grid(524288), stream=stream0)
        buf300 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1364, reinterpret_tensor(buf299, (1, 512, 1024), (0, 1024, 1), 0), out=buf300)
        del permute_1364
        buf301 = reinterpret_tensor(buf291, (1, 512, 1024), (524288, 1024, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf299, (1, 512, 1024), (0, 1024, 1), 0), permute_1379, out=buf301)
        del permute_1379
        buf305 = reinterpret_tensor(buf299, (512, 1, 1024), (1024, 524288, 1), 0); del buf299  # reuse
        buf308 = reinterpret_tensor(buf288, (512, 1, 1024), (1024, 1024, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf271, buf296, buf298, buf301, primals_312, mul_146, div_39, getitem_217, buf305, buf308, 512, 1024, grid=grid(512), stream=stream0)
        del div_39
        del getitem_217
        del primals_312
        buf306 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf307 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf271, buf296, buf298, buf301, mul_146, buf306, buf307, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_146
        buf309 = reinterpret_tensor(buf264, (512, 4096), (4096, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (512, 1024), (1024, 1), 0), permute_1384, out=buf309)
        del permute_1384
        buf310 = reinterpret_tensor(buf281, (1024, 4096), (4096, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (1024, 512), (1, 1024), 0), view_682, out=buf310)
        del view_682
        buf311 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf308, buf311, 4096, 128, grid=grid(4096), stream=stream0)
        buf312 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf311, buf312, 1024, 4, grid=grid(1024), stream=stream0)
        buf313 = reinterpret_tensor(buf309, (512, 1, 4096), (4096, 4096, 1), 0); del buf309  # reuse
        # Source Nodes: [output_139], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf313, getitem_215, addmm_34, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_34
        del getitem_215
        buf314 = reinterpret_tensor(buf308, (512, 1024), (1024, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (512, 4096), (4096, 1), 0), permute_1388, out=buf314)
        del permute_1388
        buf315 = reinterpret_tensor(buf278, (4096, 1024), (1024, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (4096, 512), (1, 4096), 0), view_680, out=buf315)
        del view_680
        buf316 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf313, buf316, 16384, 128, grid=grid(16384), stream=stream0)
        buf317 = reinterpret_tensor(buf311, (1, 4096), (4096, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf316, buf317, 4096, 4, grid=grid(4096), stream=stream0)
        buf320 = reinterpret_tensor(buf301, (512, 1, 1024), (1024, 524288, 1), 0); del buf301  # reuse
        buf323 = reinterpret_tensor(buf298, (512, 1, 1024), (1024, 1024, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf305, buf314, primals_306, mul_141, div_40, getitem_211, buf320, buf323, 512, 1024, grid=grid(512), stream=stream0)
        del div_40
        del getitem_211
        del primals_306
        buf321 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf322 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf305, buf314, mul_141, buf321, buf322, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_141
        buf324 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1393, reinterpret_tensor(buf323, (1, 512, 1024), (0, 1024, 1), 0), out=buf324)
        del permute_1393
        buf325 = reinterpret_tensor(buf314, (1, 512, 1024), (524288, 1024, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf323, (1, 512, 1024), (0, 1024, 1), 0), permute_1394, out=buf325)
        del permute_1394
        buf326 = reinterpret_tensor(buf323, (16, 512, 64), (32768, 64, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1400, reinterpret_tensor(buf325, (16, 512, 64), (1, 1024, 16), 0), out=buf326)
        del permute_1400
        buf327 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (16, 512, 64), (1, 1024, 16), 0), permute_1401, out=buf327)
        del permute_1401
        buf330 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf327, getitem_209, alias_32, buf330, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_32
        del getitem_209
        buf329 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf329, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf329, [None, None, None, iota_2], buf330, True)
        buf334 = buf287; del buf287  # reuse
        buf336 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf329, buf334, buf336, 8388608, grid=grid(8388608), stream=stream0)
        buf335 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1407, buf334, out=buf335)
        del permute_1407
        buf337 = reinterpret_tensor(buf325, (16, 512, 64), (32768, 64, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf336, permute_1408, out=buf337)
        del permute_1408
        buf338 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf337, buf338, 1024, 512, grid=grid(1024), stream=stream0)
        buf339 = reinterpret_tensor(buf305, (16, 64, 512), (32768, 512, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1414, reinterpret_tensor(buf330, (16, 512, 512), (262144, 512, 1), 0), out=buf339)
        del permute_1414
        buf340 = reinterpret_tensor(buf296, (16, 512, 64), (32768, 64, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (16, 512, 512), (262144, 512, 1), 0), permute_1415, out=buf340)
        del permute_1415
        buf341 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf340, buf341, 1024, 512, grid=grid(1024), stream=stream0)
        buf342 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf335, (1, 1024, 1024), (0, 1, 1024), 0), out=buf342)
        buf343 = reinterpret_tensor(buf271, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf326, buf343, 524288, grid=grid(524288), stream=stream0)
        buf344 = reinterpret_tensor(buf335, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1425, reinterpret_tensor(buf343, (1, 512, 1024), (0, 1024, 1), 0), out=buf344)
        buf345 = reinterpret_tensor(buf326, (1, 512, 1024), (524288, 1024, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf343, (1, 512, 1024), (0, 1024, 1), 0), permute_1426, out=buf345)
        del permute_1426
        buf346 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1425, reinterpret_tensor(buf339, (1, 512, 1024), (0, 1, 512), 0), out=buf346)
        buf347 = reinterpret_tensor(buf343, (1, 512, 1024), (524288, 1024, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf339, (1, 512, 1024), (0, 1, 512), 0), permute_1433, out=buf347)
        del permute_1433
        buf348 = reinterpret_tensor(buf339, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf337, buf340, buf348, 524288, grid=grid(524288), stream=stream0)
        buf349 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1425, reinterpret_tensor(buf348, (1, 512, 1024), (0, 1024, 1), 0), out=buf349)
        del permute_1425
        buf350 = reinterpret_tensor(buf340, (1, 512, 1024), (524288, 1024, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf348, (1, 512, 1024), (0, 1024, 1), 0), permute_1440, out=buf350)
        del permute_1440
        buf354 = reinterpret_tensor(buf348, (512, 1, 1024), (1024, 524288, 1), 0); del buf348  # reuse
        buf357 = reinterpret_tensor(buf337, (512, 1, 1024), (1024, 1024, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf320, buf345, buf347, buf350, primals_304, mul_138, div_41, getitem_205, buf354, buf357, 512, 1024, grid=grid(512), stream=stream0)
        del div_41
        del getitem_205
        del primals_304
        buf355 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf356 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf320, buf345, buf347, buf350, mul_138, buf355, buf356, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_138
        buf358 = reinterpret_tensor(buf313, (512, 4096), (4096, 1), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (512, 1024), (1024, 1), 0), permute_1445, out=buf358)
        del permute_1445
        buf359 = reinterpret_tensor(buf330, (1024, 4096), (4096, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (1024, 512), (1, 1024), 0), view_644, out=buf359)
        del view_644
        buf360 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf357, buf360, 4096, 128, grid=grid(4096), stream=stream0)
        buf361 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf360, buf361, 1024, 4, grid=grid(1024), stream=stream0)
        buf362 = reinterpret_tensor(buf358, (512, 1, 4096), (4096, 4096, 1), 0); del buf358  # reuse
        # Source Nodes: [output_131], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf362, getitem_203, addmm_32, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_32
        del getitem_203
        buf363 = reinterpret_tensor(buf357, (512, 1024), (1024, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (512, 4096), (4096, 1), 0), permute_1449, out=buf363)
        del permute_1449
        buf364 = reinterpret_tensor(buf327, (4096, 1024), (1024, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (4096, 512), (1, 4096), 0), view_642, out=buf364)
        del view_642
        buf365 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf362, buf365, 16384, 128, grid=grid(16384), stream=stream0)
        buf366 = reinterpret_tensor(buf360, (1, 4096), (4096, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf365, buf366, 4096, 4, grid=grid(4096), stream=stream0)
        buf369 = reinterpret_tensor(buf350, (512, 1, 1024), (1024, 524288, 1), 0); del buf350  # reuse
        buf372 = reinterpret_tensor(buf347, (512, 1, 1024), (1024, 1024, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf354, buf363, primals_298, mul_133, div_42, getitem_199, buf369, buf372, 512, 1024, grid=grid(512), stream=stream0)
        del div_42
        del getitem_199
        del primals_298
        buf370 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf371 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf354, buf363, mul_133, buf370, buf371, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_133
        buf373 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1454, reinterpret_tensor(buf372, (1, 512, 1024), (0, 1024, 1), 0), out=buf373)
        del permute_1454
        buf374 = reinterpret_tensor(buf363, (1, 512, 1024), (524288, 1024, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (1, 512, 1024), (0, 1024, 1), 0), permute_1455, out=buf374)
        del permute_1455
        buf375 = reinterpret_tensor(buf372, (16, 512, 64), (32768, 64, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1461, reinterpret_tensor(buf374, (16, 512, 64), (1, 1024, 16), 0), out=buf375)
        del permute_1461
        buf376 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf374, (16, 512, 64), (1, 1024, 16), 0), permute_1462, out=buf376)
        del permute_1462
        buf379 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf376, getitem_197, alias_33, buf379, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_33
        del getitem_197
        buf378 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf378, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf378, [None, None, None, iota_2], buf379, True)
        buf383 = buf336; del buf336  # reuse
        buf385 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf378, buf383, buf385, 8388608, grid=grid(8388608), stream=stream0)
        buf384 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1468, buf383, out=buf384)
        del permute_1468
        buf386 = reinterpret_tensor(buf374, (16, 512, 64), (32768, 64, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf385, permute_1469, out=buf386)
        del permute_1469
        buf387 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf386, buf387, 1024, 512, grid=grid(1024), stream=stream0)
        buf388 = reinterpret_tensor(buf354, (16, 64, 512), (32768, 512, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1475, reinterpret_tensor(buf379, (16, 512, 512), (262144, 512, 1), 0), out=buf388)
        del permute_1475
        buf389 = reinterpret_tensor(buf345, (16, 512, 64), (32768, 64, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf379, (16, 512, 512), (262144, 512, 1), 0), permute_1476, out=buf389)
        del permute_1476
        buf390 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf389, buf390, 1024, 512, grid=grid(1024), stream=stream0)
        buf391 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf384, (1, 1024, 1024), (0, 1, 1024), 0), out=buf391)
        buf392 = reinterpret_tensor(buf320, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf375, buf392, 524288, grid=grid(524288), stream=stream0)
        buf393 = reinterpret_tensor(buf384, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1486, reinterpret_tensor(buf392, (1, 512, 1024), (0, 1024, 1), 0), out=buf393)
        buf394 = reinterpret_tensor(buf375, (1, 512, 1024), (524288, 1024, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf392, (1, 512, 1024), (0, 1024, 1), 0), permute_1487, out=buf394)
        del permute_1487
        buf395 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1486, reinterpret_tensor(buf388, (1, 512, 1024), (0, 1, 512), 0), out=buf395)
        buf396 = reinterpret_tensor(buf392, (1, 512, 1024), (524288, 1024, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf388, (1, 512, 1024), (0, 1, 512), 0), permute_1494, out=buf396)
        del permute_1494
        buf397 = reinterpret_tensor(buf388, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf386, buf389, buf397, 524288, grid=grid(524288), stream=stream0)
        buf398 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1486, reinterpret_tensor(buf397, (1, 512, 1024), (0, 1024, 1), 0), out=buf398)
        del permute_1486
        buf399 = reinterpret_tensor(buf389, (1, 512, 1024), (524288, 1024, 1), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf397, (1, 512, 1024), (0, 1024, 1), 0), permute_1501, out=buf399)
        del permute_1501
        buf403 = reinterpret_tensor(buf397, (512, 1, 1024), (1024, 524288, 1), 0); del buf397  # reuse
        buf406 = reinterpret_tensor(buf386, (512, 1, 1024), (1024, 1024, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf369, buf394, buf396, buf399, primals_296, mul_130, div_43, getitem_193, buf403, buf406, 512, 1024, grid=grid(512), stream=stream0)
        del div_43
        del getitem_193
        del primals_296
        buf404 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf405 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf369, buf394, buf396, buf399, mul_130, buf404, buf405, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_130
        buf407 = reinterpret_tensor(buf362, (512, 4096), (4096, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (512, 1024), (1024, 1), 0), permute_1506, out=buf407)
        del permute_1506
        buf408 = reinterpret_tensor(buf379, (1024, 4096), (4096, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (1024, 512), (1, 1024), 0), view_606, out=buf408)
        del view_606
        buf409 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf406, buf409, 4096, 128, grid=grid(4096), stream=stream0)
        buf410 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf409, buf410, 1024, 4, grid=grid(1024), stream=stream0)
        buf411 = reinterpret_tensor(buf407, (512, 1, 4096), (4096, 4096, 1), 0); del buf407  # reuse
        # Source Nodes: [output_123], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf411, getitem_191, addmm_30, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_30
        del getitem_191
        buf412 = reinterpret_tensor(buf406, (512, 1024), (1024, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 4096), (4096, 1), 0), permute_1510, out=buf412)
        del permute_1510
        buf413 = reinterpret_tensor(buf376, (4096, 1024), (1024, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (4096, 512), (1, 4096), 0), view_604, out=buf413)
        del view_604
        buf414 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf411, buf414, 16384, 128, grid=grid(16384), stream=stream0)
        buf415 = reinterpret_tensor(buf409, (1, 4096), (4096, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf414, buf415, 4096, 4, grid=grid(4096), stream=stream0)
        buf418 = reinterpret_tensor(buf399, (512, 1, 1024), (1024, 524288, 1), 0); del buf399  # reuse
        buf421 = reinterpret_tensor(buf396, (512, 1, 1024), (1024, 1024, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf403, buf412, primals_290, mul_125, div_44, getitem_187, buf418, buf421, 512, 1024, grid=grid(512), stream=stream0)
        del div_44
        del getitem_187
        del primals_290
        buf419 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf420 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf403, buf412, mul_125, buf419, buf420, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_125
        buf422 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1515, reinterpret_tensor(buf421, (1, 512, 1024), (0, 1024, 1), 0), out=buf422)
        del permute_1515
        buf423 = reinterpret_tensor(buf412, (1, 512, 1024), (524288, 1024, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (1, 512, 1024), (0, 1024, 1), 0), permute_1516, out=buf423)
        del permute_1516
        buf424 = reinterpret_tensor(buf421, (16, 512, 64), (32768, 64, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1522, reinterpret_tensor(buf423, (16, 512, 64), (1, 1024, 16), 0), out=buf424)
        del permute_1522
        buf425 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf423, (16, 512, 64), (1, 1024, 16), 0), permute_1523, out=buf425)
        del permute_1523
        buf428 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf425, getitem_185, alias_34, buf428, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_34
        del getitem_185
        buf427 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf427, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf427, [None, None, None, iota_2], buf428, True)
        buf432 = buf385; del buf385  # reuse
        buf434 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf427, buf432, buf434, 8388608, grid=grid(8388608), stream=stream0)
        buf433 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1529, buf432, out=buf433)
        del permute_1529
        buf435 = reinterpret_tensor(buf423, (16, 512, 64), (32768, 64, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf434, permute_1530, out=buf435)
        del permute_1530
        buf436 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf435, buf436, 1024, 512, grid=grid(1024), stream=stream0)
        buf437 = reinterpret_tensor(buf403, (16, 64, 512), (32768, 512, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1536, reinterpret_tensor(buf428, (16, 512, 512), (262144, 512, 1), 0), out=buf437)
        del permute_1536
        buf438 = reinterpret_tensor(buf394, (16, 512, 64), (32768, 64, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf428, (16, 512, 512), (262144, 512, 1), 0), permute_1537, out=buf438)
        del permute_1537
        buf439 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf438, buf439, 1024, 512, grid=grid(1024), stream=stream0)
        buf440 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf433, (1, 1024, 1024), (0, 1, 1024), 0), out=buf440)
        buf441 = reinterpret_tensor(buf369, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf424, buf441, 524288, grid=grid(524288), stream=stream0)
        buf442 = reinterpret_tensor(buf433, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1547, reinterpret_tensor(buf441, (1, 512, 1024), (0, 1024, 1), 0), out=buf442)
        buf443 = reinterpret_tensor(buf424, (1, 512, 1024), (524288, 1024, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (1, 512, 1024), (0, 1024, 1), 0), permute_1548, out=buf443)
        del permute_1548
        buf444 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1547, reinterpret_tensor(buf437, (1, 512, 1024), (0, 1, 512), 0), out=buf444)
        buf445 = reinterpret_tensor(buf441, (1, 512, 1024), (524288, 1024, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf437, (1, 512, 1024), (0, 1, 512), 0), permute_1555, out=buf445)
        del permute_1555
        buf446 = reinterpret_tensor(buf437, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf435, buf438, buf446, 524288, grid=grid(524288), stream=stream0)
        buf447 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1547, reinterpret_tensor(buf446, (1, 512, 1024), (0, 1024, 1), 0), out=buf447)
        del permute_1547
        buf448 = reinterpret_tensor(buf438, (1, 512, 1024), (524288, 1024, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf446, (1, 512, 1024), (0, 1024, 1), 0), permute_1562, out=buf448)
        del permute_1562
        buf452 = reinterpret_tensor(buf446, (512, 1, 1024), (1024, 524288, 1), 0); del buf446  # reuse
        buf455 = reinterpret_tensor(buf435, (512, 1, 1024), (1024, 1024, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf418, buf443, buf445, buf448, primals_288, mul_122, div_45, getitem_181, buf452, buf455, 512, 1024, grid=grid(512), stream=stream0)
        del div_45
        del getitem_181
        del primals_288
        buf453 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf454 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf418, buf443, buf445, buf448, mul_122, buf453, buf454, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_122
        buf456 = reinterpret_tensor(buf411, (512, 4096), (4096, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (512, 1024), (1024, 1), 0), permute_1567, out=buf456)
        del permute_1567
        buf457 = reinterpret_tensor(buf428, (1024, 4096), (4096, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (1024, 512), (1, 1024), 0), view_568, out=buf457)
        del view_568
        buf458 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf455, buf458, 4096, 128, grid=grid(4096), stream=stream0)
        buf459 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf458, buf459, 1024, 4, grid=grid(1024), stream=stream0)
        buf460 = reinterpret_tensor(buf456, (512, 1, 4096), (4096, 4096, 1), 0); del buf456  # reuse
        # Source Nodes: [output_115], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf460, getitem_179, addmm_28, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_28
        del getitem_179
        buf461 = reinterpret_tensor(buf455, (512, 1024), (1024, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (512, 4096), (4096, 1), 0), permute_1571, out=buf461)
        del permute_1571
        buf462 = reinterpret_tensor(buf425, (4096, 1024), (1024, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (4096, 512), (1, 4096), 0), view_566, out=buf462)
        del view_566
        buf463 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf460, buf463, 16384, 128, grid=grid(16384), stream=stream0)
        buf464 = reinterpret_tensor(buf458, (1, 4096), (4096, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf463, buf464, 4096, 4, grid=grid(4096), stream=stream0)
        buf467 = reinterpret_tensor(buf448, (512, 1, 1024), (1024, 524288, 1), 0); del buf448  # reuse
        buf470 = reinterpret_tensor(buf445, (512, 1, 1024), (1024, 1024, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf452, buf461, primals_282, mul_117, div_46, getitem_175, buf467, buf470, 512, 1024, grid=grid(512), stream=stream0)
        del div_46
        del getitem_175
        del primals_282
        buf468 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf469 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf452, buf461, mul_117, buf468, buf469, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_117
        buf471 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1576, reinterpret_tensor(buf470, (1, 512, 1024), (0, 1024, 1), 0), out=buf471)
        del permute_1576
        buf472 = reinterpret_tensor(buf461, (1, 512, 1024), (524288, 1024, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (1, 512, 1024), (0, 1024, 1), 0), permute_1577, out=buf472)
        del permute_1577
        buf473 = reinterpret_tensor(buf470, (16, 512, 64), (32768, 64, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1583, reinterpret_tensor(buf472, (16, 512, 64), (1, 1024, 16), 0), out=buf473)
        del permute_1583
        buf474 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf472, (16, 512, 64), (1, 1024, 16), 0), permute_1584, out=buf474)
        del permute_1584
        buf477 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf474, getitem_173, alias_35, buf477, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_35
        del getitem_173
        buf476 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf476, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf476, [None, None, None, iota_2], buf477, True)
        buf481 = buf434; del buf434  # reuse
        buf483 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf476, buf481, buf483, 8388608, grid=grid(8388608), stream=stream0)
        buf482 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1590, buf481, out=buf482)
        del permute_1590
        buf484 = reinterpret_tensor(buf472, (16, 512, 64), (32768, 64, 1), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf483, permute_1591, out=buf484)
        del permute_1591
        buf485 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf484, buf485, 1024, 512, grid=grid(1024), stream=stream0)
        buf486 = reinterpret_tensor(buf452, (16, 64, 512), (32768, 512, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1597, reinterpret_tensor(buf477, (16, 512, 512), (262144, 512, 1), 0), out=buf486)
        del permute_1597
        buf487 = reinterpret_tensor(buf443, (16, 512, 64), (32768, 64, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf477, (16, 512, 512), (262144, 512, 1), 0), permute_1598, out=buf487)
        del permute_1598
        buf488 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf487, buf488, 1024, 512, grid=grid(1024), stream=stream0)
        buf489 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf482, (1, 1024, 1024), (0, 1, 1024), 0), out=buf489)
        buf490 = reinterpret_tensor(buf418, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf473, buf490, 524288, grid=grid(524288), stream=stream0)
        buf491 = reinterpret_tensor(buf482, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1608, reinterpret_tensor(buf490, (1, 512, 1024), (0, 1024, 1), 0), out=buf491)
        buf492 = reinterpret_tensor(buf473, (1, 512, 1024), (524288, 1024, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf490, (1, 512, 1024), (0, 1024, 1), 0), permute_1609, out=buf492)
        del permute_1609
        buf493 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1608, reinterpret_tensor(buf486, (1, 512, 1024), (0, 1, 512), 0), out=buf493)
        buf494 = reinterpret_tensor(buf490, (1, 512, 1024), (524288, 1024, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf486, (1, 512, 1024), (0, 1, 512), 0), permute_1616, out=buf494)
        del permute_1616
        buf495 = reinterpret_tensor(buf486, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf484, buf487, buf495, 524288, grid=grid(524288), stream=stream0)
        buf496 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1608, reinterpret_tensor(buf495, (1, 512, 1024), (0, 1024, 1), 0), out=buf496)
        del permute_1608
        buf497 = reinterpret_tensor(buf487, (1, 512, 1024), (524288, 1024, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (1, 512, 1024), (0, 1024, 1), 0), permute_1623, out=buf497)
        del permute_1623
        buf501 = reinterpret_tensor(buf495, (512, 1, 1024), (1024, 524288, 1), 0); del buf495  # reuse
        buf504 = reinterpret_tensor(buf484, (512, 1, 1024), (1024, 1024, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf467, buf492, buf494, buf497, primals_280, mul_114, div_47, getitem_169, buf501, buf504, 512, 1024, grid=grid(512), stream=stream0)
        del div_47
        del getitem_169
        del primals_280
        buf502 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf503 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf467, buf492, buf494, buf497, mul_114, buf502, buf503, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_114
        buf505 = reinterpret_tensor(buf460, (512, 4096), (4096, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (512, 1024), (1024, 1), 0), permute_1628, out=buf505)
        del permute_1628
        buf506 = reinterpret_tensor(buf477, (1024, 4096), (4096, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (1024, 512), (1, 1024), 0), view_530, out=buf506)
        del view_530
        buf507 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf504, buf507, 4096, 128, grid=grid(4096), stream=stream0)
        buf508 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf507, buf508, 1024, 4, grid=grid(1024), stream=stream0)
        buf509 = reinterpret_tensor(buf505, (512, 1, 4096), (4096, 4096, 1), 0); del buf505  # reuse
        # Source Nodes: [output_107], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf509, getitem_167, addmm_26, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_26
        del getitem_167
        buf510 = reinterpret_tensor(buf504, (512, 1024), (1024, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (512, 4096), (4096, 1), 0), permute_1632, out=buf510)
        del permute_1632
        buf511 = reinterpret_tensor(buf474, (4096, 1024), (1024, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (4096, 512), (1, 4096), 0), view_528, out=buf511)
        del view_528
        buf512 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf509, buf512, 16384, 128, grid=grid(16384), stream=stream0)
        buf513 = reinterpret_tensor(buf507, (1, 4096), (4096, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf512, buf513, 4096, 4, grid=grid(4096), stream=stream0)
        buf516 = reinterpret_tensor(buf497, (512, 1, 1024), (1024, 524288, 1), 0); del buf497  # reuse
        buf519 = reinterpret_tensor(buf494, (512, 1, 1024), (1024, 1024, 1), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf501, buf510, primals_274, mul_109, div_48, getitem_163, buf516, buf519, 512, 1024, grid=grid(512), stream=stream0)
        del div_48
        del getitem_163
        del primals_274
        buf517 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf518 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf501, buf510, mul_109, buf517, buf518, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_109
        buf520 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1637, reinterpret_tensor(buf519, (1, 512, 1024), (0, 1024, 1), 0), out=buf520)
        del permute_1637
        buf521 = reinterpret_tensor(buf510, (1, 512, 1024), (524288, 1024, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf519, (1, 512, 1024), (0, 1024, 1), 0), permute_1638, out=buf521)
        del permute_1638
        buf522 = reinterpret_tensor(buf519, (16, 512, 64), (32768, 64, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1644, reinterpret_tensor(buf521, (16, 512, 64), (1, 1024, 16), 0), out=buf522)
        del permute_1644
        buf523 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf521, (16, 512, 64), (1, 1024, 16), 0), permute_1645, out=buf523)
        del permute_1645
        buf526 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf523, getitem_161, alias_36, buf526, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_36
        del getitem_161
        buf525 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf525, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf525, [None, None, None, iota_2], buf526, True)
        buf530 = buf483; del buf483  # reuse
        buf532 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf525, buf530, buf532, 8388608, grid=grid(8388608), stream=stream0)
        buf531 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1651, buf530, out=buf531)
        del permute_1651
        buf533 = reinterpret_tensor(buf521, (16, 512, 64), (32768, 64, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf532, permute_1652, out=buf533)
        del permute_1652
        buf534 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf533, buf534, 1024, 512, grid=grid(1024), stream=stream0)
        buf535 = reinterpret_tensor(buf501, (16, 64, 512), (32768, 512, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1658, reinterpret_tensor(buf526, (16, 512, 512), (262144, 512, 1), 0), out=buf535)
        del permute_1658
        buf536 = reinterpret_tensor(buf492, (16, 512, 64), (32768, 64, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf526, (16, 512, 512), (262144, 512, 1), 0), permute_1659, out=buf536)
        del permute_1659
        buf537 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf536, buf537, 1024, 512, grid=grid(1024), stream=stream0)
        buf538 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf531, (1, 1024, 1024), (0, 1, 1024), 0), out=buf538)
        buf539 = reinterpret_tensor(buf467, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf522, buf539, 524288, grid=grid(524288), stream=stream0)
        buf540 = reinterpret_tensor(buf531, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1669, reinterpret_tensor(buf539, (1, 512, 1024), (0, 1024, 1), 0), out=buf540)
        buf541 = reinterpret_tensor(buf522, (1, 512, 1024), (524288, 1024, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf539, (1, 512, 1024), (0, 1024, 1), 0), permute_1670, out=buf541)
        del permute_1670
        buf542 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1669, reinterpret_tensor(buf535, (1, 512, 1024), (0, 1, 512), 0), out=buf542)
        buf543 = reinterpret_tensor(buf539, (1, 512, 1024), (524288, 1024, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf535, (1, 512, 1024), (0, 1, 512), 0), permute_1677, out=buf543)
        del permute_1677
        buf544 = reinterpret_tensor(buf535, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf533, buf536, buf544, 524288, grid=grid(524288), stream=stream0)
        buf545 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1669, reinterpret_tensor(buf544, (1, 512, 1024), (0, 1024, 1), 0), out=buf545)
        del permute_1669
        buf546 = reinterpret_tensor(buf536, (1, 512, 1024), (524288, 1024, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf544, (1, 512, 1024), (0, 1024, 1), 0), permute_1684, out=buf546)
        del permute_1684
        buf550 = reinterpret_tensor(buf544, (512, 1, 1024), (1024, 524288, 1), 0); del buf544  # reuse
        buf553 = reinterpret_tensor(buf533, (512, 1, 1024), (1024, 1024, 1), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf516, buf541, buf543, buf546, primals_272, mul_106, div_49, getitem_157, buf550, buf553, 512, 1024, grid=grid(512), stream=stream0)
        del div_49
        del getitem_157
        del primals_272
        buf551 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf552 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf516, buf541, buf543, buf546, mul_106, buf551, buf552, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_106
        buf554 = reinterpret_tensor(buf509, (512, 4096), (4096, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (512, 1024), (1024, 1), 0), permute_1689, out=buf554)
        del permute_1689
        buf555 = reinterpret_tensor(buf526, (1024, 4096), (4096, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (1024, 512), (1, 1024), 0), view_492, out=buf555)
        del view_492
        buf556 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf553, buf556, 4096, 128, grid=grid(4096), stream=stream0)
        buf557 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf556, buf557, 1024, 4, grid=grid(1024), stream=stream0)
        buf558 = reinterpret_tensor(buf554, (512, 1, 4096), (4096, 4096, 1), 0); del buf554  # reuse
        # Source Nodes: [output_99], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf558, getitem_155, addmm_24, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_24
        del getitem_155
        buf559 = reinterpret_tensor(buf553, (512, 1024), (1024, 1), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (512, 4096), (4096, 1), 0), permute_1693, out=buf559)
        del permute_1693
        buf560 = reinterpret_tensor(buf523, (4096, 1024), (1024, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (4096, 512), (1, 4096), 0), view_490, out=buf560)
        del view_490
        buf561 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf558, buf561, 16384, 128, grid=grid(16384), stream=stream0)
        buf562 = reinterpret_tensor(buf556, (1, 4096), (4096, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf561, buf562, 4096, 4, grid=grid(4096), stream=stream0)
        buf565 = reinterpret_tensor(buf546, (512, 1, 1024), (1024, 524288, 1), 0); del buf546  # reuse
        buf568 = reinterpret_tensor(buf543, (512, 1, 1024), (1024, 1024, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf550, buf559, primals_266, mul_101, div_50, getitem_151, buf565, buf568, 512, 1024, grid=grid(512), stream=stream0)
        del div_50
        del getitem_151
        del primals_266
        buf566 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf567 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf550, buf559, mul_101, buf566, buf567, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_101
        buf569 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1698, reinterpret_tensor(buf568, (1, 512, 1024), (0, 1024, 1), 0), out=buf569)
        del permute_1698
        buf570 = reinterpret_tensor(buf559, (1, 512, 1024), (524288, 1024, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf568, (1, 512, 1024), (0, 1024, 1), 0), permute_1699, out=buf570)
        del permute_1699
        buf571 = reinterpret_tensor(buf568, (16, 512, 64), (32768, 64, 1), 0); del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1705, reinterpret_tensor(buf570, (16, 512, 64), (1, 1024, 16), 0), out=buf571)
        del permute_1705
        buf572 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf570, (16, 512, 64), (1, 1024, 16), 0), permute_1706, out=buf572)
        del permute_1706
        buf575 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf572, getitem_149, alias_37, buf575, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_37
        del getitem_149
        buf574 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf574, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf574, [None, None, None, iota_2], buf575, True)
        buf579 = buf532; del buf532  # reuse
        buf581 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf574, buf579, buf581, 8388608, grid=grid(8388608), stream=stream0)
        buf580 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1712, buf579, out=buf580)
        del permute_1712
        buf582 = reinterpret_tensor(buf570, (16, 512, 64), (32768, 64, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf581, permute_1713, out=buf582)
        del permute_1713
        buf583 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf582, buf583, 1024, 512, grid=grid(1024), stream=stream0)
        buf584 = reinterpret_tensor(buf550, (16, 64, 512), (32768, 512, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1719, reinterpret_tensor(buf575, (16, 512, 512), (262144, 512, 1), 0), out=buf584)
        del permute_1719
        buf585 = reinterpret_tensor(buf541, (16, 512, 64), (32768, 64, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf575, (16, 512, 512), (262144, 512, 1), 0), permute_1720, out=buf585)
        del permute_1720
        buf586 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf585, buf586, 1024, 512, grid=grid(1024), stream=stream0)
        buf587 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf580, (1, 1024, 1024), (0, 1, 1024), 0), out=buf587)
        buf588 = reinterpret_tensor(buf516, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf571, buf588, 524288, grid=grid(524288), stream=stream0)
        buf589 = reinterpret_tensor(buf580, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1730, reinterpret_tensor(buf588, (1, 512, 1024), (0, 1024, 1), 0), out=buf589)
        buf590 = reinterpret_tensor(buf571, (1, 512, 1024), (524288, 1024, 1), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (1, 512, 1024), (0, 1024, 1), 0), permute_1731, out=buf590)
        del permute_1731
        buf591 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1730, reinterpret_tensor(buf584, (1, 512, 1024), (0, 1, 512), 0), out=buf591)
        buf592 = reinterpret_tensor(buf588, (1, 512, 1024), (524288, 1024, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf584, (1, 512, 1024), (0, 1, 512), 0), permute_1738, out=buf592)
        del permute_1738
        buf593 = reinterpret_tensor(buf584, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf582, buf585, buf593, 524288, grid=grid(524288), stream=stream0)
        buf594 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1730, reinterpret_tensor(buf593, (1, 512, 1024), (0, 1024, 1), 0), out=buf594)
        del permute_1730
        buf595 = reinterpret_tensor(buf585, (1, 512, 1024), (524288, 1024, 1), 0); del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf593, (1, 512, 1024), (0, 1024, 1), 0), permute_1745, out=buf595)
        del permute_1745
        buf599 = reinterpret_tensor(buf593, (512, 1, 1024), (1024, 524288, 1), 0); del buf593  # reuse
        buf602 = reinterpret_tensor(buf582, (512, 1, 1024), (1024, 1024, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf565, buf590, buf592, buf595, primals_264, mul_98, div_51, getitem_145, buf599, buf602, 512, 1024, grid=grid(512), stream=stream0)
        del div_51
        del getitem_145
        del primals_264
        buf600 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf601 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf565, buf590, buf592, buf595, mul_98, buf600, buf601, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_98
        buf603 = reinterpret_tensor(buf558, (512, 4096), (4096, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (512, 1024), (1024, 1), 0), permute_1750, out=buf603)
        del permute_1750
        buf604 = reinterpret_tensor(buf575, (1024, 4096), (4096, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (1024, 512), (1, 1024), 0), view_454, out=buf604)
        del view_454
        buf605 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf602, buf605, 4096, 128, grid=grid(4096), stream=stream0)
        buf606 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf605, buf606, 1024, 4, grid=grid(1024), stream=stream0)
        buf607 = reinterpret_tensor(buf603, (512, 1, 4096), (4096, 4096, 1), 0); del buf603  # reuse
        # Source Nodes: [output_91], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf607, getitem_143, addmm_22, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_22
        del getitem_143
        buf608 = reinterpret_tensor(buf602, (512, 1024), (1024, 1), 0); del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf607, (512, 4096), (4096, 1), 0), permute_1754, out=buf608)
        del permute_1754
        buf609 = reinterpret_tensor(buf572, (4096, 1024), (1024, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf607, (4096, 512), (1, 4096), 0), view_452, out=buf609)
        del view_452
        buf610 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf607, buf610, 16384, 128, grid=grid(16384), stream=stream0)
        buf611 = reinterpret_tensor(buf605, (1, 4096), (4096, 1), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf610, buf611, 4096, 4, grid=grid(4096), stream=stream0)
        buf614 = reinterpret_tensor(buf595, (512, 1, 1024), (1024, 524288, 1), 0); del buf595  # reuse
        buf617 = reinterpret_tensor(buf592, (512, 1, 1024), (1024, 1024, 1), 0); del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf599, buf608, primals_258, mul_93, div_52, getitem_139, buf614, buf617, 512, 1024, grid=grid(512), stream=stream0)
        del div_52
        del getitem_139
        del primals_258
        buf615 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf616 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf599, buf608, mul_93, buf615, buf616, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_93
        buf618 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1759, reinterpret_tensor(buf617, (1, 512, 1024), (0, 1024, 1), 0), out=buf618)
        del permute_1759
        buf619 = reinterpret_tensor(buf608, (1, 512, 1024), (524288, 1024, 1), 0); del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf617, (1, 512, 1024), (0, 1024, 1), 0), permute_1760, out=buf619)
        del permute_1760
        buf620 = reinterpret_tensor(buf617, (16, 512, 64), (32768, 64, 1), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1766, reinterpret_tensor(buf619, (16, 512, 64), (1, 1024, 16), 0), out=buf620)
        del permute_1766
        buf621 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf619, (16, 512, 64), (1, 1024, 16), 0), permute_1767, out=buf621)
        del permute_1767
        buf624 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf621, getitem_137, alias_38, buf624, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_38
        del getitem_137
        buf623 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf623, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf623, [None, None, None, iota_2], buf624, True)
        buf628 = buf581; del buf581  # reuse
        buf630 = buf579; del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf623, buf628, buf630, 8388608, grid=grid(8388608), stream=stream0)
        buf629 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1773, buf628, out=buf629)
        del permute_1773
        buf631 = reinterpret_tensor(buf619, (16, 512, 64), (32768, 64, 1), 0); del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf630, permute_1774, out=buf631)
        del permute_1774
        buf632 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf631, buf632, 1024, 512, grid=grid(1024), stream=stream0)
        buf633 = reinterpret_tensor(buf599, (16, 64, 512), (32768, 512, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1780, reinterpret_tensor(buf624, (16, 512, 512), (262144, 512, 1), 0), out=buf633)
        del permute_1780
        buf634 = reinterpret_tensor(buf590, (16, 512, 64), (32768, 64, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf624, (16, 512, 512), (262144, 512, 1), 0), permute_1781, out=buf634)
        del permute_1781
        buf635 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf634, buf635, 1024, 512, grid=grid(1024), stream=stream0)
        buf636 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf629, (1, 1024, 1024), (0, 1, 1024), 0), out=buf636)
        buf637 = reinterpret_tensor(buf565, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf620, buf637, 524288, grid=grid(524288), stream=stream0)
        buf638 = reinterpret_tensor(buf629, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1791, reinterpret_tensor(buf637, (1, 512, 1024), (0, 1024, 1), 0), out=buf638)
        buf639 = reinterpret_tensor(buf620, (1, 512, 1024), (524288, 1024, 1), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf637, (1, 512, 1024), (0, 1024, 1), 0), permute_1792, out=buf639)
        del permute_1792
        buf640 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1791, reinterpret_tensor(buf633, (1, 512, 1024), (0, 1, 512), 0), out=buf640)
        buf641 = reinterpret_tensor(buf637, (1, 512, 1024), (524288, 1024, 1), 0); del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf633, (1, 512, 1024), (0, 1, 512), 0), permute_1799, out=buf641)
        del permute_1799
        buf642 = reinterpret_tensor(buf633, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf631, buf634, buf642, 524288, grid=grid(524288), stream=stream0)
        buf643 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1791, reinterpret_tensor(buf642, (1, 512, 1024), (0, 1024, 1), 0), out=buf643)
        del permute_1791
        buf644 = reinterpret_tensor(buf634, (1, 512, 1024), (524288, 1024, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf642, (1, 512, 1024), (0, 1024, 1), 0), permute_1806, out=buf644)
        del permute_1806
        buf648 = reinterpret_tensor(buf642, (512, 1, 1024), (1024, 524288, 1), 0); del buf642  # reuse
        buf651 = reinterpret_tensor(buf631, (512, 1, 1024), (1024, 1024, 1), 0); del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf614, buf639, buf641, buf644, primals_256, mul_90, div_53, getitem_133, buf648, buf651, 512, 1024, grid=grid(512), stream=stream0)
        del div_53
        del getitem_133
        del primals_256
        buf649 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf650 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf614, buf639, buf641, buf644, mul_90, buf649, buf650, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_90
        buf652 = reinterpret_tensor(buf607, (512, 4096), (4096, 1), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf651, (512, 1024), (1024, 1), 0), permute_1811, out=buf652)
        del permute_1811
        buf653 = reinterpret_tensor(buf624, (1024, 4096), (4096, 1), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf651, (1024, 512), (1, 1024), 0), view_416, out=buf653)
        del view_416
        buf654 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf651, buf654, 4096, 128, grid=grid(4096), stream=stream0)
        buf655 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf654, buf655, 1024, 4, grid=grid(1024), stream=stream0)
        buf656 = reinterpret_tensor(buf652, (512, 1, 4096), (4096, 4096, 1), 0); del buf652  # reuse
        # Source Nodes: [output_83], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf656, getitem_131, addmm_20, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_20
        del getitem_131
        buf657 = reinterpret_tensor(buf651, (512, 1024), (1024, 1), 0); del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (512, 4096), (4096, 1), 0), permute_1815, out=buf657)
        del permute_1815
        buf658 = reinterpret_tensor(buf621, (4096, 1024), (1024, 1), 0); del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (4096, 512), (1, 4096), 0), view_414, out=buf658)
        del view_414
        buf659 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf656, buf659, 16384, 128, grid=grid(16384), stream=stream0)
        buf660 = reinterpret_tensor(buf654, (1, 4096), (4096, 1), 0); del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf659, buf660, 4096, 4, grid=grid(4096), stream=stream0)
        buf663 = reinterpret_tensor(buf644, (512, 1, 1024), (1024, 524288, 1), 0); del buf644  # reuse
        buf666 = reinterpret_tensor(buf641, (512, 1, 1024), (1024, 1024, 1), 0); del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf648, buf657, primals_250, mul_85, div_54, getitem_127, buf663, buf666, 512, 1024, grid=grid(512), stream=stream0)
        del div_54
        del getitem_127
        del primals_250
        buf664 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf665 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf648, buf657, mul_85, buf664, buf665, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_85
        buf667 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1820, reinterpret_tensor(buf666, (1, 512, 1024), (0, 1024, 1), 0), out=buf667)
        del permute_1820
        buf668 = reinterpret_tensor(buf657, (1, 512, 1024), (524288, 1024, 1), 0); del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf666, (1, 512, 1024), (0, 1024, 1), 0), permute_1821, out=buf668)
        del permute_1821
        buf669 = reinterpret_tensor(buf666, (16, 512, 64), (32768, 64, 1), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1827, reinterpret_tensor(buf668, (16, 512, 64), (1, 1024, 16), 0), out=buf669)
        del permute_1827
        buf670 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf668, (16, 512, 64), (1, 1024, 16), 0), permute_1828, out=buf670)
        del permute_1828
        buf673 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf670, getitem_125, alias_39, buf673, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_39
        del getitem_125
        buf672 = buf623; del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf672, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf672, [None, None, None, iota_2], buf673, True)
        buf677 = buf630; del buf630  # reuse
        buf679 = buf628; del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf672, buf677, buf679, 8388608, grid=grid(8388608), stream=stream0)
        buf678 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1834, buf677, out=buf678)
        del permute_1834
        buf680 = reinterpret_tensor(buf668, (16, 512, 64), (32768, 64, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf679, permute_1835, out=buf680)
        del permute_1835
        buf681 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf680, buf681, 1024, 512, grid=grid(1024), stream=stream0)
        buf682 = reinterpret_tensor(buf648, (16, 64, 512), (32768, 512, 1), 0); del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1841, reinterpret_tensor(buf673, (16, 512, 512), (262144, 512, 1), 0), out=buf682)
        del permute_1841
        buf683 = reinterpret_tensor(buf639, (16, 512, 64), (32768, 64, 1), 0); del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf673, (16, 512, 512), (262144, 512, 1), 0), permute_1842, out=buf683)
        del permute_1842
        buf684 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf683, buf684, 1024, 512, grid=grid(1024), stream=stream0)
        buf685 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf678, (1, 1024, 1024), (0, 1, 1024), 0), out=buf685)
        buf686 = reinterpret_tensor(buf614, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf669, buf686, 524288, grid=grid(524288), stream=stream0)
        buf687 = reinterpret_tensor(buf678, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1852, reinterpret_tensor(buf686, (1, 512, 1024), (0, 1024, 1), 0), out=buf687)
        buf688 = reinterpret_tensor(buf669, (1, 512, 1024), (524288, 1024, 1), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf686, (1, 512, 1024), (0, 1024, 1), 0), permute_1853, out=buf688)
        del permute_1853
        buf689 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1852, reinterpret_tensor(buf682, (1, 512, 1024), (0, 1, 512), 0), out=buf689)
        buf690 = reinterpret_tensor(buf686, (1, 512, 1024), (524288, 1024, 1), 0); del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf682, (1, 512, 1024), (0, 1, 512), 0), permute_1860, out=buf690)
        del permute_1860
        buf691 = reinterpret_tensor(buf682, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf680, buf683, buf691, 524288, grid=grid(524288), stream=stream0)
        buf692 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1852, reinterpret_tensor(buf691, (1, 512, 1024), (0, 1024, 1), 0), out=buf692)
        del permute_1852
        buf693 = reinterpret_tensor(buf683, (1, 512, 1024), (524288, 1024, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf691, (1, 512, 1024), (0, 1024, 1), 0), permute_1867, out=buf693)
        del permute_1867
        buf697 = reinterpret_tensor(buf691, (512, 1, 1024), (1024, 524288, 1), 0); del buf691  # reuse
        buf700 = reinterpret_tensor(buf680, (512, 1, 1024), (1024, 1024, 1), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf663, buf688, buf690, buf693, primals_248, mul_82, div_55, getitem_121, buf697, buf700, 512, 1024, grid=grid(512), stream=stream0)
        del div_55
        del getitem_121
        del primals_248
        buf698 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf699 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf663, buf688, buf690, buf693, mul_82, buf698, buf699, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_82
        buf701 = reinterpret_tensor(buf656, (512, 4096), (4096, 1), 0); del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (512, 1024), (1024, 1), 0), permute_1872, out=buf701)
        del permute_1872
        buf702 = reinterpret_tensor(buf673, (1024, 4096), (4096, 1), 0); del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (1024, 512), (1, 1024), 0), view_378, out=buf702)
        del view_378
        buf703 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf700, buf703, 4096, 128, grid=grid(4096), stream=stream0)
        buf704 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf703, buf704, 1024, 4, grid=grid(1024), stream=stream0)
        buf705 = reinterpret_tensor(buf701, (512, 1, 4096), (4096, 4096, 1), 0); del buf701  # reuse
        # Source Nodes: [output_75], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf705, getitem_119, addmm_18, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_18
        del getitem_119
        buf706 = reinterpret_tensor(buf700, (512, 1024), (1024, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (512, 4096), (4096, 1), 0), permute_1876, out=buf706)
        del permute_1876
        buf707 = reinterpret_tensor(buf670, (4096, 1024), (1024, 1), 0); del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (4096, 512), (1, 4096), 0), view_376, out=buf707)
        del view_376
        buf708 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf705, buf708, 16384, 128, grid=grid(16384), stream=stream0)
        buf709 = reinterpret_tensor(buf703, (1, 4096), (4096, 1), 0); del buf703  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf708, buf709, 4096, 4, grid=grid(4096), stream=stream0)
        buf712 = reinterpret_tensor(buf693, (512, 1, 1024), (1024, 524288, 1), 0); del buf693  # reuse
        buf715 = reinterpret_tensor(buf690, (512, 1, 1024), (1024, 1024, 1), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf697, buf706, primals_242, mul_77, div_56, getitem_115, buf712, buf715, 512, 1024, grid=grid(512), stream=stream0)
        del div_56
        del getitem_115
        del primals_242
        buf713 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf714 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf697, buf706, mul_77, buf713, buf714, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_77
        buf716 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1881, reinterpret_tensor(buf715, (1, 512, 1024), (0, 1024, 1), 0), out=buf716)
        del permute_1881
        buf717 = reinterpret_tensor(buf706, (1, 512, 1024), (524288, 1024, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf715, (1, 512, 1024), (0, 1024, 1), 0), permute_1882, out=buf717)
        del permute_1882
        buf718 = reinterpret_tensor(buf715, (16, 512, 64), (32768, 64, 1), 0); del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1888, reinterpret_tensor(buf717, (16, 512, 64), (1, 1024, 16), 0), out=buf718)
        del permute_1888
        buf719 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf717, (16, 512, 64), (1, 1024, 16), 0), permute_1889, out=buf719)
        del permute_1889
        buf722 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf719, getitem_113, alias_40, buf722, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_40
        del getitem_113
        buf721 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf721, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf721, [None, None, None, iota_2], buf722, True)
        buf726 = buf679; del buf679  # reuse
        buf728 = buf677; del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf721, buf726, buf728, 8388608, grid=grid(8388608), stream=stream0)
        buf727 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1895, buf726, out=buf727)
        del permute_1895
        buf729 = reinterpret_tensor(buf717, (16, 512, 64), (32768, 64, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf728, permute_1896, out=buf729)
        del permute_1896
        buf730 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf729, buf730, 1024, 512, grid=grid(1024), stream=stream0)
        buf731 = reinterpret_tensor(buf697, (16, 64, 512), (32768, 512, 1), 0); del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1902, reinterpret_tensor(buf722, (16, 512, 512), (262144, 512, 1), 0), out=buf731)
        del permute_1902
        buf732 = reinterpret_tensor(buf688, (16, 512, 64), (32768, 64, 1), 0); del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf722, (16, 512, 512), (262144, 512, 1), 0), permute_1903, out=buf732)
        del permute_1903
        buf733 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf732, buf733, 1024, 512, grid=grid(1024), stream=stream0)
        buf734 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf727, (1, 1024, 1024), (0, 1, 1024), 0), out=buf734)
        buf735 = reinterpret_tensor(buf663, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf718, buf735, 524288, grid=grid(524288), stream=stream0)
        buf736 = reinterpret_tensor(buf727, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1913, reinterpret_tensor(buf735, (1, 512, 1024), (0, 1024, 1), 0), out=buf736)
        buf737 = reinterpret_tensor(buf718, (1, 512, 1024), (524288, 1024, 1), 0); del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf735, (1, 512, 1024), (0, 1024, 1), 0), permute_1914, out=buf737)
        del permute_1914
        buf738 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1913, reinterpret_tensor(buf731, (1, 512, 1024), (0, 1, 512), 0), out=buf738)
        buf739 = reinterpret_tensor(buf735, (1, 512, 1024), (524288, 1024, 1), 0); del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf731, (1, 512, 1024), (0, 1, 512), 0), permute_1921, out=buf739)
        del permute_1921
        buf740 = reinterpret_tensor(buf731, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf729, buf732, buf740, 524288, grid=grid(524288), stream=stream0)
        buf741 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1913, reinterpret_tensor(buf740, (1, 512, 1024), (0, 1024, 1), 0), out=buf741)
        del permute_1913
        buf742 = reinterpret_tensor(buf732, (1, 512, 1024), (524288, 1024, 1), 0); del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf740, (1, 512, 1024), (0, 1024, 1), 0), permute_1928, out=buf742)
        del permute_1928
        buf746 = reinterpret_tensor(buf740, (512, 1, 1024), (1024, 524288, 1), 0); del buf740  # reuse
        buf749 = reinterpret_tensor(buf729, (512, 1, 1024), (1024, 1024, 1), 0); del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf712, buf737, buf739, buf742, primals_240, mul_74, div_57, getitem_109, buf746, buf749, 512, 1024, grid=grid(512), stream=stream0)
        del div_57
        del getitem_109
        del primals_240
        buf747 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf748 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf712, buf737, buf739, buf742, mul_74, buf747, buf748, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_74
        buf750 = reinterpret_tensor(buf705, (512, 4096), (4096, 1), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (512, 1024), (1024, 1), 0), permute_1933, out=buf750)
        del permute_1933
        buf751 = reinterpret_tensor(buf722, (1024, 4096), (4096, 1), 0); del buf722  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (1024, 512), (1, 1024), 0), view_340, out=buf751)
        del view_340
        buf752 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf749, buf752, 4096, 128, grid=grid(4096), stream=stream0)
        buf753 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf752, buf753, 1024, 4, grid=grid(1024), stream=stream0)
        buf754 = reinterpret_tensor(buf750, (512, 1, 4096), (4096, 4096, 1), 0); del buf750  # reuse
        # Source Nodes: [output_67], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf754, getitem_107, addmm_16, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_16
        del getitem_107
        buf755 = reinterpret_tensor(buf749, (512, 1024), (1024, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (512, 4096), (4096, 1), 0), permute_1937, out=buf755)
        del permute_1937
        buf756 = reinterpret_tensor(buf719, (4096, 1024), (1024, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (4096, 512), (1, 4096), 0), view_338, out=buf756)
        del view_338
        buf757 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf754, buf757, 16384, 128, grid=grid(16384), stream=stream0)
        buf758 = reinterpret_tensor(buf752, (1, 4096), (4096, 1), 0); del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf757, buf758, 4096, 4, grid=grid(4096), stream=stream0)
        buf761 = reinterpret_tensor(buf742, (512, 1, 1024), (1024, 524288, 1), 0); del buf742  # reuse
        buf764 = reinterpret_tensor(buf739, (512, 1, 1024), (1024, 1024, 1), 0); del buf739  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf746, buf755, primals_234, mul_69, div_58, getitem_103, buf761, buf764, 512, 1024, grid=grid(512), stream=stream0)
        del div_58
        del getitem_103
        del primals_234
        buf762 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf763 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf746, buf755, mul_69, buf762, buf763, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_69
        buf765 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1942, reinterpret_tensor(buf764, (1, 512, 1024), (0, 1024, 1), 0), out=buf765)
        del permute_1942
        buf766 = reinterpret_tensor(buf755, (1, 512, 1024), (524288, 1024, 1), 0); del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf764, (1, 512, 1024), (0, 1024, 1), 0), permute_1943, out=buf766)
        del permute_1943
        buf767 = reinterpret_tensor(buf764, (16, 512, 64), (32768, 64, 1), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1949, reinterpret_tensor(buf766, (16, 512, 64), (1, 1024, 16), 0), out=buf767)
        del permute_1949
        buf768 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf766, (16, 512, 64), (1, 1024, 16), 0), permute_1950, out=buf768)
        del permute_1950
        buf771 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf768, getitem_101, alias_41, buf771, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_41
        del getitem_101
        buf770 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf770, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf770, [None, None, None, iota_2], buf771, True)
        buf775 = buf728; del buf728  # reuse
        buf777 = buf726; del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf770, buf775, buf777, 8388608, grid=grid(8388608), stream=stream0)
        buf776 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1956, buf775, out=buf776)
        del permute_1956
        buf778 = reinterpret_tensor(buf766, (16, 512, 64), (32768, 64, 1), 0); del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf777, permute_1957, out=buf778)
        del permute_1957
        buf779 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf778, buf779, 1024, 512, grid=grid(1024), stream=stream0)
        buf780 = reinterpret_tensor(buf746, (16, 64, 512), (32768, 512, 1), 0); del buf746  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1963, reinterpret_tensor(buf771, (16, 512, 512), (262144, 512, 1), 0), out=buf780)
        del permute_1963
        buf781 = reinterpret_tensor(buf737, (16, 512, 64), (32768, 64, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf771, (16, 512, 512), (262144, 512, 1), 0), permute_1964, out=buf781)
        del permute_1964
        buf782 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf781, buf782, 1024, 512, grid=grid(1024), stream=stream0)
        buf783 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf776, (1, 1024, 1024), (0, 1, 1024), 0), out=buf783)
        buf784 = reinterpret_tensor(buf712, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf767, buf784, 524288, grid=grid(524288), stream=stream0)
        buf785 = reinterpret_tensor(buf776, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1974, reinterpret_tensor(buf784, (1, 512, 1024), (0, 1024, 1), 0), out=buf785)
        buf786 = reinterpret_tensor(buf767, (1, 512, 1024), (524288, 1024, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf784, (1, 512, 1024), (0, 1024, 1), 0), permute_1975, out=buf786)
        del permute_1975
        buf787 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1974, reinterpret_tensor(buf780, (1, 512, 1024), (0, 1, 512), 0), out=buf787)
        buf788 = reinterpret_tensor(buf784, (1, 512, 1024), (524288, 1024, 1), 0); del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf780, (1, 512, 1024), (0, 1, 512), 0), permute_1982, out=buf788)
        del permute_1982
        buf789 = reinterpret_tensor(buf780, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf778, buf781, buf789, 524288, grid=grid(524288), stream=stream0)
        buf790 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1974, reinterpret_tensor(buf789, (1, 512, 1024), (0, 1024, 1), 0), out=buf790)
        del permute_1974
        buf791 = reinterpret_tensor(buf781, (1, 512, 1024), (524288, 1024, 1), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf789, (1, 512, 1024), (0, 1024, 1), 0), permute_1989, out=buf791)
        del permute_1989
        buf795 = reinterpret_tensor(buf789, (512, 1, 1024), (1024, 524288, 1), 0); del buf789  # reuse
        buf798 = reinterpret_tensor(buf778, (512, 1, 1024), (1024, 1024, 1), 0); del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf761, buf786, buf788, buf791, primals_232, mul_66, div_59, getitem_97, buf795, buf798, 512, 1024, grid=grid(512), stream=stream0)
        del div_59
        del getitem_97
        del primals_232
        buf796 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf797 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf761, buf786, buf788, buf791, mul_66, buf796, buf797, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_66
        buf799 = reinterpret_tensor(buf754, (512, 4096), (4096, 1), 0); del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (512, 1024), (1024, 1), 0), permute_1994, out=buf799)
        del permute_1994
        buf800 = reinterpret_tensor(buf771, (1024, 4096), (4096, 1), 0); del buf771  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (1024, 512), (1, 1024), 0), view_302, out=buf800)
        del view_302
        buf801 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf798, buf801, 4096, 128, grid=grid(4096), stream=stream0)
        buf802 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf801, buf802, 1024, 4, grid=grid(1024), stream=stream0)
        buf803 = reinterpret_tensor(buf799, (512, 1, 4096), (4096, 4096, 1), 0); del buf799  # reuse
        # Source Nodes: [output_59], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf803, getitem_95, addmm_14, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_14
        del getitem_95
        buf804 = reinterpret_tensor(buf798, (512, 1024), (1024, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf803, (512, 4096), (4096, 1), 0), permute_1998, out=buf804)
        del permute_1998
        buf805 = reinterpret_tensor(buf768, (4096, 1024), (1024, 1), 0); del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf803, (4096, 512), (1, 4096), 0), view_300, out=buf805)
        del view_300
        buf806 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf803, buf806, 16384, 128, grid=grid(16384), stream=stream0)
        buf807 = reinterpret_tensor(buf801, (1, 4096), (4096, 1), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf806, buf807, 4096, 4, grid=grid(4096), stream=stream0)
        buf810 = reinterpret_tensor(buf791, (512, 1, 1024), (1024, 524288, 1), 0); del buf791  # reuse
        buf813 = reinterpret_tensor(buf788, (512, 1, 1024), (1024, 1024, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf795, buf804, primals_226, mul_61, div_60, getitem_91, buf810, buf813, 512, 1024, grid=grid(512), stream=stream0)
        del div_60
        del getitem_91
        del primals_226
        buf811 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf812 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf795, buf804, mul_61, buf811, buf812, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_61
        buf814 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2003, reinterpret_tensor(buf813, (1, 512, 1024), (0, 1024, 1), 0), out=buf814)
        del permute_2003
        buf815 = reinterpret_tensor(buf804, (1, 512, 1024), (524288, 1024, 1), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf813, (1, 512, 1024), (0, 1024, 1), 0), permute_2004, out=buf815)
        del permute_2004
        buf816 = reinterpret_tensor(buf813, (16, 512, 64), (32768, 64, 1), 0); del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2010, reinterpret_tensor(buf815, (16, 512, 64), (1, 1024, 16), 0), out=buf816)
        del permute_2010
        buf817 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf815, (16, 512, 64), (1, 1024, 16), 0), permute_2011, out=buf817)
        del permute_2011
        buf820 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf817, getitem_89, alias_42, buf820, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_42
        del getitem_89
        buf819 = buf770; del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf819, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf819, [None, None, None, iota_2], buf820, True)
        buf824 = buf777; del buf777  # reuse
        buf826 = buf775; del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf819, buf824, buf826, 8388608, grid=grid(8388608), stream=stream0)
        buf825 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2017, buf824, out=buf825)
        del permute_2017
        buf827 = reinterpret_tensor(buf815, (16, 512, 64), (32768, 64, 1), 0); del buf815  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf826, permute_2018, out=buf827)
        del permute_2018
        buf828 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf827, buf828, 1024, 512, grid=grid(1024), stream=stream0)
        buf829 = reinterpret_tensor(buf795, (16, 64, 512), (32768, 512, 1), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2024, reinterpret_tensor(buf820, (16, 512, 512), (262144, 512, 1), 0), out=buf829)
        del permute_2024
        buf830 = reinterpret_tensor(buf786, (16, 512, 64), (32768, 64, 1), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf820, (16, 512, 512), (262144, 512, 1), 0), permute_2025, out=buf830)
        del permute_2025
        buf831 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf830, buf831, 1024, 512, grid=grid(1024), stream=stream0)
        buf832 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf825, (1, 1024, 1024), (0, 1, 1024), 0), out=buf832)
        buf833 = reinterpret_tensor(buf761, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf816, buf833, 524288, grid=grid(524288), stream=stream0)
        buf834 = reinterpret_tensor(buf825, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2035, reinterpret_tensor(buf833, (1, 512, 1024), (0, 1024, 1), 0), out=buf834)
        buf835 = reinterpret_tensor(buf816, (1, 512, 1024), (524288, 1024, 1), 0); del buf816  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf833, (1, 512, 1024), (0, 1024, 1), 0), permute_2036, out=buf835)
        del permute_2036
        buf836 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2035, reinterpret_tensor(buf829, (1, 512, 1024), (0, 1, 512), 0), out=buf836)
        buf837 = reinterpret_tensor(buf833, (1, 512, 1024), (524288, 1024, 1), 0); del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf829, (1, 512, 1024), (0, 1, 512), 0), permute_2043, out=buf837)
        del permute_2043
        buf838 = reinterpret_tensor(buf829, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf829  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf827, buf830, buf838, 524288, grid=grid(524288), stream=stream0)
        buf839 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2035, reinterpret_tensor(buf838, (1, 512, 1024), (0, 1024, 1), 0), out=buf839)
        del permute_2035
        buf840 = reinterpret_tensor(buf830, (1, 512, 1024), (524288, 1024, 1), 0); del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf838, (1, 512, 1024), (0, 1024, 1), 0), permute_2050, out=buf840)
        del permute_2050
        buf844 = reinterpret_tensor(buf838, (512, 1, 1024), (1024, 524288, 1), 0); del buf838  # reuse
        buf847 = reinterpret_tensor(buf827, (512, 1, 1024), (1024, 1024, 1), 0); del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf810, buf835, buf837, buf840, primals_224, mul_58, div_61, getitem_85, buf844, buf847, 512, 1024, grid=grid(512), stream=stream0)
        del div_61
        del getitem_85
        del primals_224
        buf845 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf846 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf810, buf835, buf837, buf840, mul_58, buf845, buf846, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_58
        buf848 = reinterpret_tensor(buf803, (512, 4096), (4096, 1), 0); del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf847, (512, 1024), (1024, 1), 0), permute_2055, out=buf848)
        del permute_2055
        buf849 = reinterpret_tensor(buf820, (1024, 4096), (4096, 1), 0); del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf847, (1024, 512), (1, 1024), 0), view_264, out=buf849)
        del view_264
        buf850 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf847, buf850, 4096, 128, grid=grid(4096), stream=stream0)
        buf851 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf850, buf851, 1024, 4, grid=grid(1024), stream=stream0)
        buf852 = reinterpret_tensor(buf848, (512, 1, 4096), (4096, 4096, 1), 0); del buf848  # reuse
        # Source Nodes: [output_51], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf852, getitem_83, addmm_12, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_12
        del getitem_83
        buf853 = reinterpret_tensor(buf847, (512, 1024), (1024, 1), 0); del buf847  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf852, (512, 4096), (4096, 1), 0), permute_2059, out=buf853)
        del permute_2059
        buf854 = reinterpret_tensor(buf817, (4096, 1024), (1024, 1), 0); del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf852, (4096, 512), (1, 4096), 0), view_262, out=buf854)
        del view_262
        buf855 = buf806; del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf852, buf855, 16384, 128, grid=grid(16384), stream=stream0)
        buf856 = reinterpret_tensor(buf850, (1, 4096), (4096, 1), 0); del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf855, buf856, 4096, 4, grid=grid(4096), stream=stream0)
        buf859 = reinterpret_tensor(buf840, (512, 1, 1024), (1024, 524288, 1), 0); del buf840  # reuse
        buf862 = reinterpret_tensor(buf837, (512, 1, 1024), (1024, 1024, 1), 0); del buf837  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf844, buf853, primals_218, mul_53, div_62, getitem_79, buf859, buf862, 512, 1024, grid=grid(512), stream=stream0)
        del div_62
        del getitem_79
        del primals_218
        buf860 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf861 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf844, buf853, mul_53, buf860, buf861, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_53
        buf863 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2064, reinterpret_tensor(buf862, (1, 512, 1024), (0, 1024, 1), 0), out=buf863)
        del permute_2064
        buf864 = reinterpret_tensor(buf853, (1, 512, 1024), (524288, 1024, 1), 0); del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf862, (1, 512, 1024), (0, 1024, 1), 0), permute_2065, out=buf864)
        del permute_2065
        buf865 = reinterpret_tensor(buf862, (16, 512, 64), (32768, 64, 1), 0); del buf862  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2071, reinterpret_tensor(buf864, (16, 512, 64), (1, 1024, 16), 0), out=buf865)
        del permute_2071
        buf866 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf864, (16, 512, 64), (1, 1024, 16), 0), permute_2072, out=buf866)
        del permute_2072
        buf869 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf866, getitem_77, alias_43, buf869, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_43
        del getitem_77
        buf868 = buf819; del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf868, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf868, [None, None, None, iota_2], buf869, True)
        buf873 = buf826; del buf826  # reuse
        buf875 = buf824; del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf868, buf873, buf875, 8388608, grid=grid(8388608), stream=stream0)
        buf874 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2078, buf873, out=buf874)
        del permute_2078
        buf876 = reinterpret_tensor(buf864, (16, 512, 64), (32768, 64, 1), 0); del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf875, permute_2079, out=buf876)
        del permute_2079
        buf877 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf876, buf877, 1024, 512, grid=grid(1024), stream=stream0)
        buf878 = reinterpret_tensor(buf844, (16, 64, 512), (32768, 512, 1), 0); del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2085, reinterpret_tensor(buf869, (16, 512, 512), (262144, 512, 1), 0), out=buf878)
        del permute_2085
        buf879 = reinterpret_tensor(buf835, (16, 512, 64), (32768, 64, 1), 0); del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf869, (16, 512, 512), (262144, 512, 1), 0), permute_2086, out=buf879)
        del permute_2086
        buf880 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf879, buf880, 1024, 512, grid=grid(1024), stream=stream0)
        buf881 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf874, (1, 1024, 1024), (0, 1, 1024), 0), out=buf881)
        buf882 = reinterpret_tensor(buf810, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf810  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf865, buf882, 524288, grid=grid(524288), stream=stream0)
        buf883 = reinterpret_tensor(buf874, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2096, reinterpret_tensor(buf882, (1, 512, 1024), (0, 1024, 1), 0), out=buf883)
        buf884 = reinterpret_tensor(buf865, (1, 512, 1024), (524288, 1024, 1), 0); del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf882, (1, 512, 1024), (0, 1024, 1), 0), permute_2097, out=buf884)
        del permute_2097
        buf885 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2096, reinterpret_tensor(buf878, (1, 512, 1024), (0, 1, 512), 0), out=buf885)
        buf886 = reinterpret_tensor(buf882, (1, 512, 1024), (524288, 1024, 1), 0); del buf882  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf878, (1, 512, 1024), (0, 1, 512), 0), permute_2104, out=buf886)
        del permute_2104
        buf887 = reinterpret_tensor(buf878, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf876, buf879, buf887, 524288, grid=grid(524288), stream=stream0)
        buf888 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2096, reinterpret_tensor(buf887, (1, 512, 1024), (0, 1024, 1), 0), out=buf888)
        del permute_2096
        buf889 = reinterpret_tensor(buf879, (1, 512, 1024), (524288, 1024, 1), 0); del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf887, (1, 512, 1024), (0, 1024, 1), 0), permute_2111, out=buf889)
        del permute_2111
        buf893 = reinterpret_tensor(buf887, (512, 1, 1024), (1024, 524288, 1), 0); del buf887  # reuse
        buf896 = reinterpret_tensor(buf876, (512, 1, 1024), (1024, 1024, 1), 0); del buf876  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf859, buf884, buf886, buf889, primals_216, mul_50, div_63, getitem_73, buf893, buf896, 512, 1024, grid=grid(512), stream=stream0)
        del div_63
        del getitem_73
        del primals_216
        buf894 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf895 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf859, buf884, buf886, buf889, mul_50, buf894, buf895, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_50
        buf897 = reinterpret_tensor(buf852, (512, 4096), (4096, 1), 0); del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf896, (512, 1024), (1024, 1), 0), permute_2116, out=buf897)
        del permute_2116
        buf898 = reinterpret_tensor(buf869, (1024, 4096), (4096, 1), 0); del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf896, (1024, 512), (1, 1024), 0), view_226, out=buf898)
        del view_226
        buf899 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf896, buf899, 4096, 128, grid=grid(4096), stream=stream0)
        buf900 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf899, buf900, 1024, 4, grid=grid(1024), stream=stream0)
        buf901 = reinterpret_tensor(buf897, (512, 1, 4096), (4096, 4096, 1), 0); del buf897  # reuse
        # Source Nodes: [output_43], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf901, getitem_71, addmm_10, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_10
        del getitem_71
        buf902 = reinterpret_tensor(buf896, (512, 1024), (1024, 1), 0); del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf901, (512, 4096), (4096, 1), 0), permute_2120, out=buf902)
        del permute_2120
        buf903 = reinterpret_tensor(buf866, (4096, 1024), (1024, 1), 0); del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf901, (4096, 512), (1, 4096), 0), view_224, out=buf903)
        del view_224
        buf904 = buf855; del buf855  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf901, buf904, 16384, 128, grid=grid(16384), stream=stream0)
        buf905 = reinterpret_tensor(buf899, (1, 4096), (4096, 1), 0); del buf899  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf904, buf905, 4096, 4, grid=grid(4096), stream=stream0)
        buf908 = reinterpret_tensor(buf889, (512, 1, 1024), (1024, 524288, 1), 0); del buf889  # reuse
        buf911 = reinterpret_tensor(buf886, (512, 1, 1024), (1024, 1024, 1), 0); del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf893, buf902, primals_210, mul_45, div_64, getitem_67, buf908, buf911, 512, 1024, grid=grid(512), stream=stream0)
        del div_64
        del getitem_67
        del primals_210
        buf909 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf910 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf893, buf902, mul_45, buf909, buf910, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_45
        buf912 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2125, reinterpret_tensor(buf911, (1, 512, 1024), (0, 1024, 1), 0), out=buf912)
        del permute_2125
        buf913 = reinterpret_tensor(buf902, (1, 512, 1024), (524288, 1024, 1), 0); del buf902  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf911, (1, 512, 1024), (0, 1024, 1), 0), permute_2126, out=buf913)
        del permute_2126
        buf914 = reinterpret_tensor(buf911, (16, 512, 64), (32768, 64, 1), 0); del buf911  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2132, reinterpret_tensor(buf913, (16, 512, 64), (1, 1024, 16), 0), out=buf914)
        del permute_2132
        buf915 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf913, (16, 512, 64), (1, 1024, 16), 0), permute_2133, out=buf915)
        del permute_2133
        buf918 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf915, getitem_65, alias_44, buf918, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_44
        del getitem_65
        buf917 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf917, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf917, [None, None, None, iota_2], buf918, True)
        buf922 = buf875; del buf875  # reuse
        buf924 = buf873; del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf917, buf922, buf924, 8388608, grid=grid(8388608), stream=stream0)
        buf923 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2139, buf922, out=buf923)
        del permute_2139
        buf925 = reinterpret_tensor(buf913, (16, 512, 64), (32768, 64, 1), 0); del buf913  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf924, permute_2140, out=buf925)
        del permute_2140
        buf926 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf925, buf926, 1024, 512, grid=grid(1024), stream=stream0)
        buf927 = reinterpret_tensor(buf893, (16, 64, 512), (32768, 512, 1), 0); del buf893  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2146, reinterpret_tensor(buf918, (16, 512, 512), (262144, 512, 1), 0), out=buf927)
        del permute_2146
        buf928 = reinterpret_tensor(buf884, (16, 512, 64), (32768, 64, 1), 0); del buf884  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf918, (16, 512, 512), (262144, 512, 1), 0), permute_2147, out=buf928)
        del permute_2147
        buf929 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf928, buf929, 1024, 512, grid=grid(1024), stream=stream0)
        buf930 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf923, (1, 1024, 1024), (0, 1, 1024), 0), out=buf930)
        buf931 = reinterpret_tensor(buf859, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf914, buf931, 524288, grid=grid(524288), stream=stream0)
        buf932 = reinterpret_tensor(buf923, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf923  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2157, reinterpret_tensor(buf931, (1, 512, 1024), (0, 1024, 1), 0), out=buf932)
        buf933 = reinterpret_tensor(buf914, (1, 512, 1024), (524288, 1024, 1), 0); del buf914  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf931, (1, 512, 1024), (0, 1024, 1), 0), permute_2158, out=buf933)
        del permute_2158
        buf934 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2157, reinterpret_tensor(buf927, (1, 512, 1024), (0, 1, 512), 0), out=buf934)
        buf935 = reinterpret_tensor(buf931, (1, 512, 1024), (524288, 1024, 1), 0); del buf931  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf927, (1, 512, 1024), (0, 1, 512), 0), permute_2165, out=buf935)
        del permute_2165
        buf936 = reinterpret_tensor(buf927, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf925, buf928, buf936, 524288, grid=grid(524288), stream=stream0)
        buf937 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2157, reinterpret_tensor(buf936, (1, 512, 1024), (0, 1024, 1), 0), out=buf937)
        del permute_2157
        buf938 = reinterpret_tensor(buf928, (1, 512, 1024), (524288, 1024, 1), 0); del buf928  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf936, (1, 512, 1024), (0, 1024, 1), 0), permute_2172, out=buf938)
        del permute_2172
        buf942 = reinterpret_tensor(buf936, (512, 1, 1024), (1024, 524288, 1), 0); del buf936  # reuse
        buf945 = reinterpret_tensor(buf925, (512, 1, 1024), (1024, 1024, 1), 0); del buf925  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf908, buf933, buf935, buf938, primals_208, mul_42, div_65, getitem_61, buf942, buf945, 512, 1024, grid=grid(512), stream=stream0)
        del div_65
        del getitem_61
        del primals_208
        buf943 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf944 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf908, buf933, buf935, buf938, mul_42, buf943, buf944, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_42
        buf946 = reinterpret_tensor(buf901, (512, 4096), (4096, 1), 0); del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (512, 1024), (1024, 1), 0), permute_2177, out=buf946)
        del permute_2177
        buf947 = reinterpret_tensor(buf918, (1024, 4096), (4096, 1), 0); del buf918  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf945, (1024, 512), (1, 1024), 0), view_188, out=buf947)
        del view_188
        buf948 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf945, buf948, 4096, 128, grid=grid(4096), stream=stream0)
        buf949 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf948, buf949, 1024, 4, grid=grid(1024), stream=stream0)
        buf950 = reinterpret_tensor(buf946, (512, 1, 4096), (4096, 4096, 1), 0); del buf946  # reuse
        # Source Nodes: [output_35], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf950, getitem_59, addmm_8, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_8
        del getitem_59
        buf951 = reinterpret_tensor(buf945, (512, 1024), (1024, 1), 0); del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf950, (512, 4096), (4096, 1), 0), permute_2181, out=buf951)
        del permute_2181
        buf952 = reinterpret_tensor(buf915, (4096, 1024), (1024, 1), 0); del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf950, (4096, 512), (1, 4096), 0), view_186, out=buf952)
        del view_186
        buf953 = buf904; del buf904  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf950, buf953, 16384, 128, grid=grid(16384), stream=stream0)
        buf954 = reinterpret_tensor(buf948, (1, 4096), (4096, 1), 0); del buf948  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf953, buf954, 4096, 4, grid=grid(4096), stream=stream0)
        buf957 = reinterpret_tensor(buf938, (512, 1, 1024), (1024, 524288, 1), 0); del buf938  # reuse
        buf960 = reinterpret_tensor(buf935, (512, 1, 1024), (1024, 1024, 1), 0); del buf935  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf942, buf951, primals_202, mul_37, div_66, getitem_55, buf957, buf960, 512, 1024, grid=grid(512), stream=stream0)
        del div_66
        del getitem_55
        del primals_202
        buf958 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf959 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf942, buf951, mul_37, buf958, buf959, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_37
        buf961 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2186, reinterpret_tensor(buf960, (1, 512, 1024), (0, 1024, 1), 0), out=buf961)
        del permute_2186
        buf962 = reinterpret_tensor(buf951, (1, 512, 1024), (524288, 1024, 1), 0); del buf951  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf960, (1, 512, 1024), (0, 1024, 1), 0), permute_2187, out=buf962)
        del permute_2187
        buf963 = reinterpret_tensor(buf960, (16, 512, 64), (32768, 64, 1), 0); del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2193, reinterpret_tensor(buf962, (16, 512, 64), (1, 1024, 16), 0), out=buf963)
        del permute_2193
        buf964 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf962, (16, 512, 64), (1, 1024, 16), 0), permute_2194, out=buf964)
        del permute_2194
        buf967 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf964, getitem_53, alias_45, buf967, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_45
        del getitem_53
        buf966 = buf917; del buf917  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf966, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf966, [None, None, None, iota_2], buf967, True)
        buf971 = buf924; del buf924  # reuse
        buf973 = buf922; del buf922  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf966, buf971, buf973, 8388608, grid=grid(8388608), stream=stream0)
        buf972 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2200, buf971, out=buf972)
        del permute_2200
        buf974 = reinterpret_tensor(buf962, (16, 512, 64), (32768, 64, 1), 0); del buf962  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf973, permute_2201, out=buf974)
        del permute_2201
        buf975 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf974, buf975, 1024, 512, grid=grid(1024), stream=stream0)
        buf976 = reinterpret_tensor(buf942, (16, 64, 512), (32768, 512, 1), 0); del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2207, reinterpret_tensor(buf967, (16, 512, 512), (262144, 512, 1), 0), out=buf976)
        del permute_2207
        buf977 = reinterpret_tensor(buf933, (16, 512, 64), (32768, 64, 1), 0); del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf967, (16, 512, 512), (262144, 512, 1), 0), permute_2208, out=buf977)
        del permute_2208
        buf978 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf977, buf978, 1024, 512, grid=grid(1024), stream=stream0)
        buf979 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf972, (1, 1024, 1024), (0, 1, 1024), 0), out=buf979)
        buf980 = reinterpret_tensor(buf908, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf908  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf963, buf980, 524288, grid=grid(524288), stream=stream0)
        buf981 = reinterpret_tensor(buf972, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2218, reinterpret_tensor(buf980, (1, 512, 1024), (0, 1024, 1), 0), out=buf981)
        buf982 = reinterpret_tensor(buf963, (1, 512, 1024), (524288, 1024, 1), 0); del buf963  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf980, (1, 512, 1024), (0, 1024, 1), 0), permute_2219, out=buf982)
        del permute_2219
        buf983 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2218, reinterpret_tensor(buf976, (1, 512, 1024), (0, 1, 512), 0), out=buf983)
        buf984 = reinterpret_tensor(buf980, (1, 512, 1024), (524288, 1024, 1), 0); del buf980  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf976, (1, 512, 1024), (0, 1, 512), 0), permute_2226, out=buf984)
        del permute_2226
        buf985 = reinterpret_tensor(buf976, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf976  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf974, buf977, buf985, 524288, grid=grid(524288), stream=stream0)
        buf986 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2218, reinterpret_tensor(buf985, (1, 512, 1024), (0, 1024, 1), 0), out=buf986)
        del permute_2218
        buf987 = reinterpret_tensor(buf977, (1, 512, 1024), (524288, 1024, 1), 0); del buf977  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf985, (1, 512, 1024), (0, 1024, 1), 0), permute_2233, out=buf987)
        del permute_2233
        buf991 = reinterpret_tensor(buf985, (512, 1, 1024), (1024, 524288, 1), 0); del buf985  # reuse
        buf994 = reinterpret_tensor(buf974, (512, 1, 1024), (1024, 1024, 1), 0); del buf974  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf957, buf982, buf984, buf987, primals_200, mul_34, div_67, getitem_49, buf991, buf994, 512, 1024, grid=grid(512), stream=stream0)
        del div_67
        del getitem_49
        del primals_200
        buf992 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf993 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf957, buf982, buf984, buf987, mul_34, buf992, buf993, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_34
        buf995 = reinterpret_tensor(buf950, (512, 4096), (4096, 1), 0); del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf994, (512, 1024), (1024, 1), 0), permute_2238, out=buf995)
        del permute_2238
        buf996 = reinterpret_tensor(buf967, (1024, 4096), (4096, 1), 0); del buf967  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf994, (1024, 512), (1, 1024), 0), view_150, out=buf996)
        del view_150
        buf997 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf994, buf997, 4096, 128, grid=grid(4096), stream=stream0)
        buf998 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf997, buf998, 1024, 4, grid=grid(1024), stream=stream0)
        buf999 = reinterpret_tensor(buf995, (512, 1, 4096), (4096, 4096, 1), 0); del buf995  # reuse
        # Source Nodes: [output_27], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf999, getitem_47, addmm_6, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_6
        del getitem_47
        buf1000 = reinterpret_tensor(buf994, (512, 1024), (1024, 1), 0); del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf999, (512, 4096), (4096, 1), 0), permute_2242, out=buf1000)
        del permute_2242
        buf1001 = reinterpret_tensor(buf964, (4096, 1024), (1024, 1), 0); del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf999, (4096, 512), (1, 4096), 0), view_148, out=buf1001)
        del view_148
        buf1002 = buf953; del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf999, buf1002, 16384, 128, grid=grid(16384), stream=stream0)
        buf1003 = reinterpret_tensor(buf997, (1, 4096), (4096, 1), 0); del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf1002, buf1003, 4096, 4, grid=grid(4096), stream=stream0)
        buf1006 = reinterpret_tensor(buf987, (512, 1, 1024), (1024, 524288, 1), 0); del buf987  # reuse
        buf1009 = reinterpret_tensor(buf984, (512, 1, 1024), (1024, 1024, 1), 0); del buf984  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf991, buf1000, primals_194, mul_29, div_68, getitem_43, buf1006, buf1009, 512, 1024, grid=grid(512), stream=stream0)
        del div_68
        del getitem_43
        del primals_194
        buf1007 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1008 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf991, buf1000, mul_29, buf1007, buf1008, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_29
        buf1010 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2247, reinterpret_tensor(buf1009, (1, 512, 1024), (0, 1024, 1), 0), out=buf1010)
        del permute_2247
        buf1011 = reinterpret_tensor(buf991, (1, 512, 1024), (524288, 1024, 1), 0); del buf991  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1009, (1, 512, 1024), (0, 1024, 1), 0), permute_2248, out=buf1011)
        del permute_2248
        buf1012 = reinterpret_tensor(buf1009, (16, 512, 64), (32768, 64, 1), 0); del buf1009  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2254, reinterpret_tensor(buf1011, (16, 512, 64), (1, 1024, 16), 0), out=buf1012)
        del permute_2254
        buf1013 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1011, (16, 512, 64), (1, 1024, 16), 0), permute_2255, out=buf1013)
        del permute_2255
        buf1016 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf1013, getitem_41, alias_46, buf1016, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_46
        del getitem_41
        buf1015 = buf966; del buf966  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf1015, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf1015, [None, None, None, iota_2], buf1016, True)
        buf1020 = buf973; del buf973  # reuse
        buf1022 = buf971; del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf1015, buf1020, buf1022, 8388608, grid=grid(8388608), stream=stream0)
        buf1021 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2261, buf1020, out=buf1021)
        del permute_2261
        buf1023 = reinterpret_tensor(buf1011, (16, 512, 64), (32768, 64, 1), 0); del buf1011  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1022, permute_2262, out=buf1023)
        del permute_2262
        buf1024 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1023, buf1024, 1024, 512, grid=grid(1024), stream=stream0)
        buf1025 = reinterpret_tensor(buf1000, (16, 64, 512), (32768, 512, 1), 0); del buf1000  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2268, reinterpret_tensor(buf1016, (16, 512, 512), (262144, 512, 1), 0), out=buf1025)
        del permute_2268
        buf1026 = reinterpret_tensor(buf982, (16, 512, 64), (32768, 64, 1), 0); del buf982  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1016, (16, 512, 512), (262144, 512, 1), 0), permute_2269, out=buf1026)
        del permute_2269
        buf1027 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1026, buf1027, 1024, 512, grid=grid(1024), stream=stream0)
        buf1028 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf1021, (1, 1024, 1024), (0, 1, 1024), 0), out=buf1028)
        buf1029 = reinterpret_tensor(buf957, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf957  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf1012, buf1029, 524288, grid=grid(524288), stream=stream0)
        buf1030 = reinterpret_tensor(buf1021, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf1021  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2279, reinterpret_tensor(buf1029, (1, 512, 1024), (0, 1024, 1), 0), out=buf1030)
        buf1031 = reinterpret_tensor(buf1012, (1, 512, 1024), (524288, 1024, 1), 0); del buf1012  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1029, (1, 512, 1024), (0, 1024, 1), 0), permute_2280, out=buf1031)
        del permute_2280
        buf1032 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2279, reinterpret_tensor(buf1025, (1, 512, 1024), (0, 1, 512), 0), out=buf1032)
        buf1033 = reinterpret_tensor(buf1029, (1, 512, 1024), (524288, 1024, 1), 0); del buf1029  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1025, (1, 512, 1024), (0, 1, 512), 0), permute_2287, out=buf1033)
        del permute_2287
        buf1034 = reinterpret_tensor(buf1025, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf1025  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf1023, buf1026, buf1034, 524288, grid=grid(524288), stream=stream0)
        buf1035 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2279, reinterpret_tensor(buf1034, (1, 512, 1024), (0, 1024, 1), 0), out=buf1035)
        del permute_2279
        buf1036 = reinterpret_tensor(buf1026, (1, 512, 1024), (524288, 1024, 1), 0); del buf1026  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1034, (1, 512, 1024), (0, 1024, 1), 0), permute_2294, out=buf1036)
        del permute_2294
        buf1040 = reinterpret_tensor(buf1034, (512, 1, 1024), (1024, 524288, 1), 0); del buf1034  # reuse
        buf1043 = reinterpret_tensor(buf1023, (512, 1, 1024), (1024, 1024, 1), 0); del buf1023  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf1006, buf1031, buf1033, buf1036, primals_192, mul_26, div_69, getitem_37, buf1040, buf1043, 512, 1024, grid=grid(512), stream=stream0)
        del div_69
        del getitem_37
        del primals_192
        buf1041 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1042 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf1006, buf1031, buf1033, buf1036, mul_26, buf1041, buf1042, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_26
        buf1044 = reinterpret_tensor(buf999, (512, 4096), (4096, 1), 0); del buf999  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (512, 1024), (1024, 1), 0), permute_2299, out=buf1044)
        del permute_2299
        buf1045 = reinterpret_tensor(buf1016, (1024, 4096), (4096, 1), 0); del buf1016  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (1024, 512), (1, 1024), 0), view_112, out=buf1045)
        del view_112
        buf1046 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf1043, buf1046, 4096, 128, grid=grid(4096), stream=stream0)
        buf1047 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf1046, buf1047, 1024, 4, grid=grid(1024), stream=stream0)
        buf1048 = reinterpret_tensor(buf1044, (512, 1, 4096), (4096, 4096, 1), 0); del buf1044  # reuse
        # Source Nodes: [output_19], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf1048, getitem_35, addmm_4, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_4
        del getitem_35
        buf1049 = reinterpret_tensor(buf1043, (512, 1024), (1024, 1), 0); del buf1043  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1048, (512, 4096), (4096, 1), 0), permute_2303, out=buf1049)
        del permute_2303
        buf1050 = reinterpret_tensor(buf1013, (4096, 1024), (1024, 1), 0); del buf1013  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1048, (4096, 512), (1, 4096), 0), view_110, out=buf1050)
        del view_110
        buf1051 = buf1002; del buf1002  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1048, buf1051, 16384, 128, grid=grid(16384), stream=stream0)
        buf1052 = reinterpret_tensor(buf1046, (1, 4096), (4096, 1), 0); del buf1046  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf1051, buf1052, 4096, 4, grid=grid(4096), stream=stream0)
        buf1055 = reinterpret_tensor(buf1036, (512, 1, 1024), (1024, 524288, 1), 0); del buf1036  # reuse
        buf1058 = reinterpret_tensor(buf1033, (512, 1, 1024), (1024, 1024, 1), 0); del buf1033  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf1040, buf1049, primals_186, mul_21, div_70, getitem_31, buf1055, buf1058, 512, 1024, grid=grid(512), stream=stream0)
        del div_70
        del getitem_31
        del primals_186
        buf1056 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1057 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf1040, buf1049, mul_21, buf1056, buf1057, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_21
        buf1059 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2308, reinterpret_tensor(buf1058, (1, 512, 1024), (0, 1024, 1), 0), out=buf1059)
        del permute_2308
        buf1060 = reinterpret_tensor(buf1049, (1, 512, 1024), (524288, 1024, 1), 0); del buf1049  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1058, (1, 512, 1024), (0, 1024, 1), 0), permute_2309, out=buf1060)
        del permute_2309
        buf1061 = reinterpret_tensor(buf1058, (16, 512, 64), (32768, 64, 1), 0); del buf1058  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2315, reinterpret_tensor(buf1060, (16, 512, 64), (1, 1024, 16), 0), out=buf1061)
        del permute_2315
        buf1062 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1060, (16, 512, 64), (1, 1024, 16), 0), permute_2316, out=buf1062)
        del permute_2316
        buf1065 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf1062, getitem_29, alias_47, buf1065, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_47
        del getitem_29
        buf1064 = buf1015; del buf1015  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf1064, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf1064, [None, None, None, iota_2], buf1065, True)
        buf1069 = buf1022; del buf1022  # reuse
        buf1071 = buf1020; del buf1020  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf1064, buf1069, buf1071, 8388608, grid=grid(8388608), stream=stream0)
        buf1070 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2322, buf1069, out=buf1070)
        del permute_2322
        buf1072 = reinterpret_tensor(buf1060, (16, 512, 64), (32768, 64, 1), 0); del buf1060  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1071, permute_2323, out=buf1072)
        del permute_2323
        buf1073 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1072, buf1073, 1024, 512, grid=grid(1024), stream=stream0)
        buf1074 = reinterpret_tensor(buf1040, (16, 64, 512), (32768, 512, 1), 0); del buf1040  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2329, reinterpret_tensor(buf1065, (16, 512, 512), (262144, 512, 1), 0), out=buf1074)
        del permute_2329
        buf1075 = reinterpret_tensor(buf1031, (16, 512, 64), (32768, 64, 1), 0); del buf1031  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1065, (16, 512, 512), (262144, 512, 1), 0), permute_2330, out=buf1075)
        del permute_2330
        buf1076 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1075, buf1076, 1024, 512, grid=grid(1024), stream=stream0)
        buf1077 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf1070, (1, 1024, 1024), (0, 1, 1024), 0), out=buf1077)
        buf1078 = reinterpret_tensor(buf1006, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf1006  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf1061, buf1078, 524288, grid=grid(524288), stream=stream0)
        buf1079 = reinterpret_tensor(buf1070, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf1070  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2340, reinterpret_tensor(buf1078, (1, 512, 1024), (0, 1024, 1), 0), out=buf1079)
        buf1080 = reinterpret_tensor(buf1061, (1, 512, 1024), (524288, 1024, 1), 0); del buf1061  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1078, (1, 512, 1024), (0, 1024, 1), 0), permute_2341, out=buf1080)
        del permute_2341
        buf1081 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2340, reinterpret_tensor(buf1074, (1, 512, 1024), (0, 1, 512), 0), out=buf1081)
        buf1082 = reinterpret_tensor(buf1078, (1, 512, 1024), (524288, 1024, 1), 0); del buf1078  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1074, (1, 512, 1024), (0, 1, 512), 0), permute_2348, out=buf1082)
        del permute_2348
        buf1083 = reinterpret_tensor(buf1074, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf1074  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf1072, buf1075, buf1083, 524288, grid=grid(524288), stream=stream0)
        buf1084 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2340, reinterpret_tensor(buf1083, (1, 512, 1024), (0, 1024, 1), 0), out=buf1084)
        del permute_2340
        buf1085 = reinterpret_tensor(buf1075, (1, 512, 1024), (524288, 1024, 1), 0); del buf1075  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1083, (1, 512, 1024), (0, 1024, 1), 0), permute_2355, out=buf1085)
        del permute_2355
        buf1089 = reinterpret_tensor(buf1083, (512, 1, 1024), (1024, 524288, 1), 0); del buf1083  # reuse
        buf1092 = reinterpret_tensor(buf1072, (512, 1, 1024), (1024, 1024, 1), 0); del buf1072  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf1055, buf1080, buf1082, buf1085, primals_184, mul_18, div_71, getitem_25, buf1089, buf1092, 512, 1024, grid=grid(512), stream=stream0)
        del div_71
        del getitem_25
        del primals_184
        buf1090 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1091 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf1055, buf1080, buf1082, buf1085, mul_18, buf1090, buf1091, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_18
        buf1093 = reinterpret_tensor(buf1048, (512, 4096), (4096, 1), 0); del buf1048  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1092, (512, 1024), (1024, 1), 0), permute_2360, out=buf1093)
        del permute_2360
        buf1094 = reinterpret_tensor(buf1065, (1024, 4096), (4096, 1), 0); del buf1065  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1092, (1024, 512), (1, 1024), 0), view_74, out=buf1094)
        del view_74
        buf1095 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf1092, buf1095, 4096, 128, grid=grid(4096), stream=stream0)
        buf1096 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf1095, buf1096, 1024, 4, grid=grid(1024), stream=stream0)
        buf1097 = reinterpret_tensor(buf1093, (512, 1, 4096), (4096, 4096, 1), 0); del buf1093  # reuse
        # Source Nodes: [output_11], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf1097, getitem_23, addmm_2, 2097152, grid=grid(2097152), stream=stream0)
        del addmm_2
        del getitem_23
        buf1098 = reinterpret_tensor(buf1092, (512, 1024), (1024, 1), 0); del buf1092  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1097, (512, 4096), (4096, 1), 0), permute_2364, out=buf1098)
        del permute_2364
        buf1099 = reinterpret_tensor(buf1062, (4096, 1024), (1024, 1), 0); del buf1062  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1097, (4096, 512), (1, 4096), 0), view_72, out=buf1099)
        del view_72
        buf1100 = buf1051; del buf1051  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1097, buf1100, 16384, 128, grid=grid(16384), stream=stream0)
        buf1101 = reinterpret_tensor(buf1095, (1, 4096), (4096, 1), 0); del buf1095  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf1100, buf1101, 4096, 4, grid=grid(4096), stream=stream0)
        buf1104 = reinterpret_tensor(buf1085, (512, 1, 1024), (1024, 524288, 1), 0); del buf1085  # reuse
        buf1107 = reinterpret_tensor(buf1082, (512, 1, 1024), (1024, 1024, 1), 0); del buf1082  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf1089, buf1098, primals_178, mul_13, div_72, getitem_19, buf1104, buf1107, 512, 1024, grid=grid(512), stream=stream0)
        del div_72
        del getitem_19
        del primals_178
        buf1105 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1106 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf1089, buf1098, mul_13, buf1105, buf1106, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_13
        buf1108 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2369, reinterpret_tensor(buf1107, (1, 512, 1024), (0, 1024, 1), 0), out=buf1108)
        del permute_2369
        buf1109 = reinterpret_tensor(buf1098, (1, 512, 1024), (524288, 1024, 1), 0); del buf1098  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1107, (1, 512, 1024), (0, 1024, 1), 0), permute_2370, out=buf1109)
        del permute_2370
        buf1110 = reinterpret_tensor(buf1107, (16, 512, 64), (32768, 64, 1), 0); del buf1107  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2376, reinterpret_tensor(buf1109, (16, 512, 64), (1, 1024, 16), 0), out=buf1110)
        del permute_2376
        buf1111 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1109, (16, 512, 64), (1, 1024, 16), 0), permute_2377, out=buf1111)
        del permute_2377
        buf1114 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf1111, getitem_17, alias_48, buf1114, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_48
        del getitem_17
        buf1113 = buf1064; del buf1064  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf1113, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf1113, [None, None, None, iota_2], buf1114, True)
        buf1118 = buf1071; del buf1071  # reuse
        buf1120 = buf1069; del buf1069  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf1113, buf1118, buf1120, 8388608, grid=grid(8388608), stream=stream0)
        buf1119 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2383, buf1118, out=buf1119)
        del permute_2383
        buf1121 = reinterpret_tensor(buf1109, (16, 512, 64), (32768, 64, 1), 0); del buf1109  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1120, permute_2384, out=buf1121)
        del permute_2384
        buf1122 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1121, buf1122, 1024, 512, grid=grid(1024), stream=stream0)
        buf1123 = reinterpret_tensor(buf1089, (16, 64, 512), (32768, 512, 1), 0); del buf1089  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2390, reinterpret_tensor(buf1114, (16, 512, 512), (262144, 512, 1), 0), out=buf1123)
        del permute_2390
        buf1124 = reinterpret_tensor(buf1080, (16, 512, 64), (32768, 64, 1), 0); del buf1080  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1114, (16, 512, 512), (262144, 512, 1), 0), permute_2391, out=buf1124)
        del permute_2391
        buf1125 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1124, buf1125, 1024, 512, grid=grid(1024), stream=stream0)
        buf1126 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf1119, (1, 1024, 1024), (0, 1, 1024), 0), out=buf1126)
        buf1127 = reinterpret_tensor(buf1055, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf1055  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf1110, buf1127, 524288, grid=grid(524288), stream=stream0)
        buf1128 = reinterpret_tensor(buf1119, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf1119  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2401, reinterpret_tensor(buf1127, (1, 512, 1024), (0, 1024, 1), 0), out=buf1128)
        buf1129 = reinterpret_tensor(buf1110, (1, 512, 1024), (524288, 1024, 1), 0); del buf1110  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1127, (1, 512, 1024), (0, 1024, 1), 0), permute_2402, out=buf1129)
        del permute_2402
        buf1130 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2401, reinterpret_tensor(buf1123, (1, 512, 1024), (0, 1, 512), 0), out=buf1130)
        buf1131 = reinterpret_tensor(buf1127, (1, 512, 1024), (524288, 1024, 1), 0); del buf1127  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1123, (1, 512, 1024), (0, 1, 512), 0), permute_2409, out=buf1131)
        del permute_2409
        buf1132 = reinterpret_tensor(buf1123, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf1123  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf1121, buf1124, buf1132, 524288, grid=grid(524288), stream=stream0)
        buf1133 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2401, reinterpret_tensor(buf1132, (1, 512, 1024), (0, 1024, 1), 0), out=buf1133)
        del permute_2401
        buf1134 = reinterpret_tensor(buf1124, (1, 512, 1024), (524288, 1024, 1), 0); del buf1124  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1132, (1, 512, 1024), (0, 1024, 1), 0), permute_2416, out=buf1134)
        del permute_2416
        buf1138 = reinterpret_tensor(buf1132, (512, 1, 1024), (1024, 524288, 1), 0); del buf1132  # reuse
        buf1141 = reinterpret_tensor(buf1121, (512, 1, 1024), (1024, 1024, 1), 0); del buf1121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf1104, buf1129, buf1131, buf1134, primals_176, mul_10, div_73, getitem_13, buf1138, buf1141, 512, 1024, grid=grid(512), stream=stream0)
        del div_73
        del getitem_13
        del primals_176
        buf1139 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1140 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf1104, buf1129, buf1131, buf1134, mul_10, buf1139, buf1140, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_10
        buf1142 = reinterpret_tensor(buf1097, (512, 4096), (4096, 1), 0); del buf1097  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1141, (512, 1024), (1024, 1), 0), permute_2421, out=buf1142)
        del permute_2421
        buf1143 = reinterpret_tensor(buf1114, (1024, 4096), (4096, 1), 0); del buf1114  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1141, (1024, 512), (1, 1024), 0), view_36, out=buf1143)
        del view_36
        buf1144 = empty_strided((1, 1024, 4), (4096, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf1141, buf1144, 4096, 128, grid=grid(4096), stream=stream0)
        buf1145 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf1144, buf1145, 1024, 4, grid=grid(1024), stream=stream0)
        buf1146 = reinterpret_tensor(buf1142, (512, 1, 4096), (4096, 4096, 1), 0); del buf1142  # reuse
        # Source Nodes: [output_3], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_dropout_backward]
        triton_poi_fused_gelu_gelu_backward_native_dropout_backward_8.run(buf1146, getitem_11, addmm, 2097152, grid=grid(2097152), stream=stream0)
        del addmm
        del getitem_11
        buf1147 = reinterpret_tensor(buf1141, (512, 1024), (1024, 1), 0); del buf1141  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1146, (512, 4096), (4096, 1), 0), permute_2425, out=buf1147)
        del permute_2425
        buf1148 = reinterpret_tensor(buf1111, (4096, 1024), (1024, 1), 0); del buf1111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1146, (4096, 512), (1, 4096), 0), view_34, out=buf1148)
        del view_34
        buf1149 = buf1100; del buf1100  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf1146, buf1149, 16384, 128, grid=grid(16384), stream=stream0)
        del buf1146
        buf1150 = reinterpret_tensor(buf1144, (1, 4096), (4096, 1), 0); del buf1144  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf1149, buf1150, 4096, 4, grid=grid(4096), stream=stream0)
        del buf1149
        buf1153 = reinterpret_tensor(buf1134, (512, 1, 1024), (1024, 524288, 1), 0); del buf1134  # reuse
        buf1156 = reinterpret_tensor(buf1131, (512, 1, 1024), (1024, 1024, 1), 0); del buf1131  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf1138, buf1147, primals_170, mul_5, div_74, getitem_7, buf1153, buf1156, 512, 1024, grid=grid(512), stream=stream0)
        del div_74
        del getitem_7
        del primals_170
        buf1154 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1155 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf1138, buf1147, mul_5, buf1154, buf1155, 1024, 512, grid=grid(1024), stream=stream0)
        del mul_5
        buf1157 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2430, reinterpret_tensor(buf1156, (1, 512, 1024), (0, 1024, 1), 0), out=buf1157)
        del permute_2430
        buf1158 = reinterpret_tensor(buf1147, (1, 512, 1024), (524288, 1024, 1), 0); del buf1147  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1156, (1, 512, 1024), (0, 1024, 1), 0), permute_2431, out=buf1158)
        del permute_2431
        buf1159 = reinterpret_tensor(buf1156, (16, 512, 64), (32768, 64, 1), 0); del buf1156  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2437, reinterpret_tensor(buf1158, (16, 512, 64), (1, 1024, 16), 0), out=buf1159)
        del permute_2437
        buf1160 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1158, (16, 512, 64), (1, 1024, 16), 0), permute_2438, out=buf1160)
        del permute_2438
        buf1163 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_mul_native_dropout_backward_13.run(buf1160, getitem_5, alias_49, buf1163, 8192, 512, grid=grid(8192), stream=stream0)
        del alias_49
        del buf1160
        del getitem_5
        buf1162 = buf1113; del buf1113  # reuse
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14.run(buf1162, 8380416, grid=grid(8380416), stream=stream0)
        aten.index_put_(buf1162, [None, None, None, iota_2], buf1163, True)
        del iota_2
        buf1167 = buf1120; del buf1120  # reuse
        buf1169 = buf1118; del buf1118  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm, aten.slice_backward]
        triton_poi_fused_bmm_slice_backward_15.run(buf1162, buf1167, buf1169, 8388608, grid=grid(8388608), stream=stream0)
        del buf1162
        buf1168 = empty((16, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2444, buf1167, out=buf1168)
        del buf1167
        del permute_2444
        buf1170 = reinterpret_tensor(buf1158, (16, 512, 64), (32768, 64, 1), 0); del buf1158  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1169, permute_2445, out=buf1170)
        del buf1169
        del permute_2445
        buf1171 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1170, buf1171, 1024, 512, grid=grid(1024), stream=stream0)
        buf1172 = reinterpret_tensor(buf1138, (16, 64, 512), (32768, 512, 1), 0); del buf1138  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2451, reinterpret_tensor(buf1163, (16, 512, 512), (262144, 512, 1), 0), out=buf1172)
        del permute_2451
        buf1173 = reinterpret_tensor(buf1129, (16, 512, 64), (32768, 64, 1), 0); del buf1129  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1163, (16, 512, 512), (262144, 512, 1), 0), permute_2452, out=buf1173)
        del buf1163
        del permute_2452
        buf1174 = empty((1, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf1173, buf1174, 1024, 512, grid=grid(1024), stream=stream0)
        buf1175 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1055, reinterpret_tensor(buf1168, (1, 1024, 1024), (0, 1, 1024), 0), out=buf1175)
        del permute_1055
        buf1176 = reinterpret_tensor(buf1104, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf1104  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf1159, buf1176, 524288, grid=grid(524288), stream=stream0)
        buf1177 = reinterpret_tensor(buf1168, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf1168  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2462, reinterpret_tensor(buf1176, (1, 512, 1024), (0, 1024, 1), 0), out=buf1177)
        buf1178 = reinterpret_tensor(buf1159, (1, 512, 1024), (524288, 1024, 1), 0); del buf1159  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1176, (1, 512, 1024), (0, 1024, 1), 0), permute_2463, out=buf1178)
        del permute_2463
        buf1179 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2462, reinterpret_tensor(buf1172, (1, 512, 1024), (0, 1, 512), 0), out=buf1179)
        buf1180 = reinterpret_tensor(buf1176, (1, 512, 1024), (524288, 1024, 1), 0); del buf1176  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1172, (1, 512, 1024), (0, 1, 512), 0), permute_2470, out=buf1180)
        del permute_2470
        buf1181 = reinterpret_tensor(buf1172, (512, 1, 1, 16, 64), (1024, 1024, 1024, 64, 1), 0); del buf1172  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf1170, buf1173, buf1181, 524288, grid=grid(524288), stream=stream0)
        del buf1170
        buf1182 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2462, reinterpret_tensor(buf1181, (1, 512, 1024), (0, 1024, 1), 0), out=buf1182)
        del permute_2462
        buf1183 = reinterpret_tensor(buf1173, (1, 512, 1024), (524288, 1024, 1), 0); del buf1173  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1181, (1, 512, 1024), (0, 1024, 1), 0), permute_2477, out=buf1183)
        del buf1181
        del permute_2477
        buf1184 = empty((32000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_21.run(buf1184, 32768000, grid=grid(32768000), stream=stream0)
        buf1185 = buf1153; del buf1153  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_nll_loss_forward_22.run(buf1185, permute, buf1178, buf1180, buf1183, getitem_1, 524288, grid=grid(524288), stream=stream0)
        del buf1178
        del buf1180
        del buf1183
        del getitem_1
        aten.index_put_(buf1184, [permute], buf1185, True)
        del buf1185
        del permute
        return (reinterpret_tensor(buf1182, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1179, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1177, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1175, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1174, (16, 64), (64, 1), 0), reinterpret_tensor(buf1171, (16, 64), (64, 1), 0), reinterpret_tensor(buf1157, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf1133, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1130, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1128, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1126, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1125, (16, 64), (64, 1), 0), reinterpret_tensor(buf1122, (16, 64), (64, 1), 0), reinterpret_tensor(buf1108, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf1084, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1081, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1079, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1077, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1076, (16, 64), (64, 1), 0), reinterpret_tensor(buf1073, (16, 64), (64, 1), 0), reinterpret_tensor(buf1059, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf1035, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1032, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1030, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1028, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf1027, (16, 64), (64, 1), 0), reinterpret_tensor(buf1024, (16, 64), (64, 1), 0), reinterpret_tensor(buf1010, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf986, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf983, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf981, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf979, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf978, (16, 64), (64, 1), 0), reinterpret_tensor(buf975, (16, 64), (64, 1), 0), reinterpret_tensor(buf961, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf937, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf934, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf932, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf930, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf929, (16, 64), (64, 1), 0), reinterpret_tensor(buf926, (16, 64), (64, 1), 0), reinterpret_tensor(buf912, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf888, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf885, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf883, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf881, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf880, (16, 64), (64, 1), 0), reinterpret_tensor(buf877, (16, 64), (64, 1), 0), reinterpret_tensor(buf863, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf839, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf836, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf834, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf832, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf831, (16, 64), (64, 1), 0), reinterpret_tensor(buf828, (16, 64), (64, 1), 0), reinterpret_tensor(buf814, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf790, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf787, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf785, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf783, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf782, (16, 64), (64, 1), 0), reinterpret_tensor(buf779, (16, 64), (64, 1), 0), reinterpret_tensor(buf765, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf741, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf738, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf736, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf734, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf733, (16, 64), (64, 1), 0), reinterpret_tensor(buf730, (16, 64), (64, 1), 0), reinterpret_tensor(buf716, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf692, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf689, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf687, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf685, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf684, (16, 64), (64, 1), 0), reinterpret_tensor(buf681, (16, 64), (64, 1), 0), reinterpret_tensor(buf667, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf643, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf640, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf638, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf636, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf635, (16, 64), (64, 1), 0), reinterpret_tensor(buf632, (16, 64), (64, 1), 0), reinterpret_tensor(buf618, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf594, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf591, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf589, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf587, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf586, (16, 64), (64, 1), 0), reinterpret_tensor(buf583, (16, 64), (64, 1), 0), reinterpret_tensor(buf569, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf545, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf542, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf540, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf538, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf537, (16, 64), (64, 1), 0), reinterpret_tensor(buf534, (16, 64), (64, 1), 0), reinterpret_tensor(buf520, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf496, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf493, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf491, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf489, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf488, (16, 64), (64, 1), 0), reinterpret_tensor(buf485, (16, 64), (64, 1), 0), reinterpret_tensor(buf471, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf447, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf444, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf442, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf440, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf439, (16, 64), (64, 1), 0), reinterpret_tensor(buf436, (16, 64), (64, 1), 0), reinterpret_tensor(buf422, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf398, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf395, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf393, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf391, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf390, (16, 64), (64, 1), 0), reinterpret_tensor(buf387, (16, 64), (64, 1), 0), reinterpret_tensor(buf373, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf349, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf346, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf344, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf342, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf341, (16, 64), (64, 1), 0), reinterpret_tensor(buf338, (16, 64), (64, 1), 0), reinterpret_tensor(buf324, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf300, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf297, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf295, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf293, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf292, (16, 64), (64, 1), 0), reinterpret_tensor(buf289, (16, 64), (64, 1), 0), reinterpret_tensor(buf275, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf251, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf248, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf246, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf244, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf243, (16, 64), (64, 1), 0), reinterpret_tensor(buf240, (16, 64), (64, 1), 0), reinterpret_tensor(buf226, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf202, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf199, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf197, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf195, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf194, (16, 64), (64, 1), 0), reinterpret_tensor(buf191, (16, 64), (64, 1), 0), reinterpret_tensor(buf177, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf153, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf150, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf148, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf146, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf145, (16, 64), (64, 1), 0), reinterpret_tensor(buf142, (16, 64), (64, 1), 0), reinterpret_tensor(buf128, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf104, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf101, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf99, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf97, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf96, (16, 64), (64, 1), 0), reinterpret_tensor(buf93, (16, 64), (64, 1), 0), reinterpret_tensor(buf79, (1024, 16, 64), (1, 1024, 16384), 0), reinterpret_tensor(buf55, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf52, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf50, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf48, (1024, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf47, (16, 64), (64, 1), 0), reinterpret_tensor(buf44, (16, 64), (64, 1), 0), reinterpret_tensor(buf30, (1024, 16, 64), (1, 1024, 16384), 0), buf1184, buf1154, buf1155, reinterpret_tensor(buf1148, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1150, (4096, ), (1, ), 0), reinterpret_tensor(buf1143, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1145, (1024, ), (1, ), 0), buf1139, buf1140, buf1105, buf1106, reinterpret_tensor(buf1099, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1101, (4096, ), (1, ), 0), reinterpret_tensor(buf1094, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1096, (1024, ), (1, ), 0), buf1090, buf1091, buf1056, buf1057, reinterpret_tensor(buf1050, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1052, (4096, ), (1, ), 0), reinterpret_tensor(buf1045, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf1047, (1024, ), (1, ), 0), buf1041, buf1042, buf1007, buf1008, reinterpret_tensor(buf1001, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1003, (4096, ), (1, ), 0), reinterpret_tensor(buf996, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf998, (1024, ), (1, ), 0), buf992, buf993, buf958, buf959, reinterpret_tensor(buf952, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf954, (4096, ), (1, ), 0), reinterpret_tensor(buf947, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf949, (1024, ), (1, ), 0), buf943, buf944, buf909, buf910, reinterpret_tensor(buf903, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf905, (4096, ), (1, ), 0), reinterpret_tensor(buf898, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf900, (1024, ), (1, ), 0), buf894, buf895, buf860, buf861, reinterpret_tensor(buf854, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf856, (4096, ), (1, ), 0), reinterpret_tensor(buf849, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf851, (1024, ), (1, ), 0), buf845, buf846, buf811, buf812, reinterpret_tensor(buf805, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf807, (4096, ), (1, ), 0), reinterpret_tensor(buf800, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf802, (1024, ), (1, ), 0), buf796, buf797, buf762, buf763, reinterpret_tensor(buf756, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf758, (4096, ), (1, ), 0), reinterpret_tensor(buf751, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf753, (1024, ), (1, ), 0), buf747, buf748, buf713, buf714, reinterpret_tensor(buf707, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf709, (4096, ), (1, ), 0), reinterpret_tensor(buf702, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf704, (1024, ), (1, ), 0), buf698, buf699, buf664, buf665, reinterpret_tensor(buf658, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf660, (4096, ), (1, ), 0), reinterpret_tensor(buf653, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf655, (1024, ), (1, ), 0), buf649, buf650, buf615, buf616, reinterpret_tensor(buf609, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf611, (4096, ), (1, ), 0), reinterpret_tensor(buf604, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf606, (1024, ), (1, ), 0), buf600, buf601, buf566, buf567, reinterpret_tensor(buf560, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf562, (4096, ), (1, ), 0), reinterpret_tensor(buf555, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf557, (1024, ), (1, ), 0), buf551, buf552, buf517, buf518, reinterpret_tensor(buf511, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf513, (4096, ), (1, ), 0), reinterpret_tensor(buf506, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf508, (1024, ), (1, ), 0), buf502, buf503, buf468, buf469, reinterpret_tensor(buf462, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf464, (4096, ), (1, ), 0), reinterpret_tensor(buf457, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf459, (1024, ), (1, ), 0), buf453, buf454, buf419, buf420, reinterpret_tensor(buf413, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf415, (4096, ), (1, ), 0), reinterpret_tensor(buf408, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf410, (1024, ), (1, ), 0), buf404, buf405, buf370, buf371, reinterpret_tensor(buf364, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf366, (4096, ), (1, ), 0), reinterpret_tensor(buf359, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf361, (1024, ), (1, ), 0), buf355, buf356, buf321, buf322, reinterpret_tensor(buf315, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf317, (4096, ), (1, ), 0), reinterpret_tensor(buf310, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf312, (1024, ), (1, ), 0), buf306, buf307, buf272, buf273, reinterpret_tensor(buf266, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf268, (4096, ), (1, ), 0), reinterpret_tensor(buf261, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf263, (1024, ), (1, ), 0), buf257, buf258, buf223, buf224, reinterpret_tensor(buf217, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf219, (4096, ), (1, ), 0), reinterpret_tensor(buf212, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf214, (1024, ), (1, ), 0), buf208, buf209, buf174, buf175, reinterpret_tensor(buf168, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf170, (4096, ), (1, ), 0), reinterpret_tensor(buf163, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf165, (1024, ), (1, ), 0), buf159, buf160, buf125, buf126, reinterpret_tensor(buf119, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf121, (4096, ), (1, ), 0), reinterpret_tensor(buf114, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf116, (1024, ), (1, ), 0), buf110, buf111, buf76, buf77, reinterpret_tensor(buf70, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf72, (4096, ), (1, ), 0), reinterpret_tensor(buf65, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf67, (1024, ), (1, ), 0), buf61, buf62, buf27, buf28, reinterpret_tensor(buf21, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf23, (4096, ), (1, ), 0), reinterpret_tensor(buf16, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf18, (1024, ), (1, ), 0), buf12, buf13, reinterpret_tensor(buf7, (32000, 1024), (1024, 1), 0), reinterpret_tensor(buf8, (32000, ), (1, ), 0), None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_170 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    permute = rand_strided((512, 1), (1, 512), device='cuda:0', dtype=torch.int64)
    getitem_1 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    iota_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.int64)
    getitem_5 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_7 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_5 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_36 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_19 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_13 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_23 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_74 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_18 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_31 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_21 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_112 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_26 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_43 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_29 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_150 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_34 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_55 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_37 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_186 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_188 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_42 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_65 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_67 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_45 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_224 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_226 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_50 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_79 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_53 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_12 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_264 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_58 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_89 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_91 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_61 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_300 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_302 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_66 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_103 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_69 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_338 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_340 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_74 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_115 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_77 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_376 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_119 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_378 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_82 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_127 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_85 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_414 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_20 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_416 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_133 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_90 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_137 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_139 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_93 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_452 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_143 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_454 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_145 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_98 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_151 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_101 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_490 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_24 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_492 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_106 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_161 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_163 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_109 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_528 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_167 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_530 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_169 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_114 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_173 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_175 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_117 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_566 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_179 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_568 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_181 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_122 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_185 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_187 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_125 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_604 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_606 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_193 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_130 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_197 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_199 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_133 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_642 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_32 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_203 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_644 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_205 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_138 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_209 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_211 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_141 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_680 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_215 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_682 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_217 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_146 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_221 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_223 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_149 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_718 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_36 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_227 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_720 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_229 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_154 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_233 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_235 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_157 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_756 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_239 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_758 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_241 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_162 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_245 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_247 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_165 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_794 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_251 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_796 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_253 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_170 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_257 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_259 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_173 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_832 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_263 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_834 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_265 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_178 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_269 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_271 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_181 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_870 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_44 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_275 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_872 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_277 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_186 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_281 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_283 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_189 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_908 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_287 = rand_strided((512, 1, 4096), (4096, 4096, 1), device='cuda:0', dtype=torch.bool)
    view_910 = rand_strided((512, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_289 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_194 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_293 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_912 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    sub_73 = rand_strided((512, 32000), (32000, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_5 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_1013 = rand_strided((32000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1018 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1022 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1027 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1028 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1034 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1035 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_26 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1041 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1042 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1048 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1049 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1055 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1059 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1060 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1067 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1074 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1079 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1083 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1088 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1089 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1095 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1096 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1102 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1103 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1109 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1110 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1120 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1121 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1128 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1135 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1140 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1144 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1149 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1150 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1156 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1157 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_28 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1163 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1164 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1170 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1171 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1181 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1182 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1189 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1196 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1201 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1205 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1210 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1211 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1217 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1218 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1224 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1225 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1231 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1232 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1242 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1243 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1250 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1257 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1262 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1266 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1271 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1272 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1278 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1279 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_30 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1285 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1286 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1292 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1293 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1303 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1304 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1311 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1318 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1323 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1327 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1332 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1333 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1339 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1340 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1346 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1347 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1353 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1354 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1364 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1365 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1372 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1379 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1384 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1388 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1393 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1394 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1400 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1401 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_32 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1407 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1408 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1414 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1415 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1425 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1426 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1433 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1440 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1445 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1449 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1454 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1455 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1461 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1462 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1468 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1469 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1475 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1476 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1486 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1487 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1494 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1501 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1506 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1510 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1515 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1516 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1522 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1523 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_34 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1529 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1530 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1536 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1537 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1547 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1548 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1555 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1562 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1567 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1571 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1576 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1577 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1583 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1584 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1590 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1591 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1597 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1598 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1608 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1609 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1616 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1623 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1628 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1632 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1637 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1638 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1644 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1645 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_36 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1651 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1652 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1658 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1659 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1669 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1670 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1677 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1684 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1689 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1693 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1698 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1699 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1705 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1706 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1712 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1713 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1719 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1720 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1730 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1731 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1738 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1745 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1750 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1754 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1759 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1760 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1766 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1767 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_38 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1773 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1774 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1780 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1781 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1791 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1792 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1799 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1806 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1811 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1815 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1820 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1821 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1827 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1828 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1834 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1835 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1841 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1842 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1852 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1853 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1860 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1867 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1872 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1876 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1881 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1882 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1888 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1889 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_40 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1895 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1896 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1902 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1903 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1913 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1914 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1921 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1928 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1933 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1937 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1942 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1943 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1949 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1950 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_1956 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1957 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1963 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1964 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1974 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1975 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1982 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_1989 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1994 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_1998 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2003 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2004 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2010 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2011 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_42 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2017 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2018 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2024 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2025 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2035 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2036 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2043 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2050 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2055 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_2059 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_62 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2064 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2065 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2071 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2072 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2078 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2079 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2085 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2086 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2096 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2097 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2104 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2111 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2116 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_2120 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2125 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2126 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2132 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2133 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_44 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2139 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2140 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2146 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2147 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2157 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2158 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2165 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2172 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_65 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2177 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_2181 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2186 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2187 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2193 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2194 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2200 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2201 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2207 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2208 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2218 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2219 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2226 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2233 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2238 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_2242 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_68 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2247 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2248 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2254 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2255 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_46 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2261 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2262 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2268 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2269 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2279 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2280 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2287 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2294 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2299 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_2303 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2308 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2309 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2315 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2316 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2322 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2323 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2329 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2330 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2340 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2341 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2348 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2355 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_71 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2360 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_2364 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2369 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2370 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2376 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2377 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_48 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2383 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2384 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2390 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2391 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2401 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2402 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2409 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2416 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2421 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_2425 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_74 = rand_strided((512, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_2430 = rand_strided((1, 1024, 512), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2431 = rand_strided((1, 1024, 1024), (0, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2437 = rand_strided((16, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_2438 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    alias_49 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_2444 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2445 = rand_strided((16, 1024, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2451 = rand_strided((16, 64, 512), (64, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2452 = rand_strided((16, 512, 64), (64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_2462 = rand_strided((1, 1024, 512), (524288, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2463 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2470 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_2477 = rand_strided((1, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 32000), (16384000, 32000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_170, primals_176, primals_178, primals_184, primals_186, primals_192, primals_194, primals_200, primals_202, primals_208, primals_210, primals_216, primals_218, primals_224, primals_226, primals_232, primals_234, primals_240, primals_242, primals_248, primals_250, primals_256, primals_258, primals_264, primals_266, primals_272, primals_274, primals_280, primals_282, primals_288, primals_290, primals_296, primals_298, primals_304, primals_306, primals_312, primals_314, primals_320, primals_322, primals_328, primals_330, primals_336, primals_338, primals_344, primals_346, primals_352, primals_354, primals_360, primals_365, permute, getitem_1, iota_2, getitem_5, getitem_7, mul_5, view_34, addmm, getitem_11, view_36, getitem_13, mul_10, getitem_17, getitem_19, mul_13, view_72, addmm_2, getitem_23, view_74, getitem_25, mul_18, getitem_29, getitem_31, mul_21, view_110, addmm_4, getitem_35, view_112, getitem_37, mul_26, getitem_41, getitem_43, mul_29, view_148, addmm_6, getitem_47, view_150, getitem_49, mul_34, getitem_53, getitem_55, mul_37, view_186, addmm_8, getitem_59, view_188, getitem_61, mul_42, getitem_65, getitem_67, mul_45, view_224, addmm_10, getitem_71, view_226, getitem_73, mul_50, getitem_77, getitem_79, mul_53, view_262, addmm_12, getitem_83, view_264, getitem_85, mul_58, getitem_89, getitem_91, mul_61, view_300, addmm_14, getitem_95, view_302, getitem_97, mul_66, getitem_101, getitem_103, mul_69, view_338, addmm_16, getitem_107, view_340, getitem_109, mul_74, getitem_113, getitem_115, mul_77, view_376, addmm_18, getitem_119, view_378, getitem_121, mul_82, getitem_125, getitem_127, mul_85, view_414, addmm_20, getitem_131, view_416, getitem_133, mul_90, getitem_137, getitem_139, mul_93, view_452, addmm_22, getitem_143, view_454, getitem_145, mul_98, getitem_149, getitem_151, mul_101, view_490, addmm_24, getitem_155, view_492, getitem_157, mul_106, getitem_161, getitem_163, mul_109, view_528, addmm_26, getitem_167, view_530, getitem_169, mul_114, getitem_173, getitem_175, mul_117, view_566, addmm_28, getitem_179, view_568, getitem_181, mul_122, getitem_185, getitem_187, mul_125, view_604, addmm_30, getitem_191, view_606, getitem_193, mul_130, getitem_197, getitem_199, mul_133, view_642, addmm_32, getitem_203, view_644, getitem_205, mul_138, getitem_209, getitem_211, mul_141, view_680, addmm_34, getitem_215, view_682, getitem_217, mul_146, getitem_221, getitem_223, mul_149, view_718, addmm_36, getitem_227, view_720, getitem_229, mul_154, getitem_233, getitem_235, mul_157, view_756, addmm_38, getitem_239, view_758, getitem_241, mul_162, getitem_245, getitem_247, mul_165, view_794, addmm_40, getitem_251, view_796, getitem_253, mul_170, getitem_257, getitem_259, mul_173, view_832, addmm_42, getitem_263, view_834, getitem_265, mul_178, getitem_269, getitem_271, mul_181, view_870, addmm_44, getitem_275, view_872, getitem_277, mul_186, getitem_281, getitem_283, mul_189, view_908, addmm_46, getitem_287, view_910, getitem_289, mul_194, getitem_293, view_912, sub_73, convert_element_type_5, permute_1013, div_27, permute_1018, permute_1022, div_28, permute_1027, permute_1028, permute_1034, permute_1035, alias_26, permute_1041, permute_1042, permute_1048, permute_1049, permute_1055, permute_1059, permute_1060, permute_1067, permute_1074, div_29, permute_1079, permute_1083, div_30, permute_1088, permute_1089, permute_1095, permute_1096, alias_27, permute_1102, permute_1103, permute_1109, permute_1110, permute_1120, permute_1121, permute_1128, permute_1135, div_31, permute_1140, permute_1144, div_32, permute_1149, permute_1150, permute_1156, permute_1157, alias_28, permute_1163, permute_1164, permute_1170, permute_1171, permute_1181, permute_1182, permute_1189, permute_1196, div_33, permute_1201, permute_1205, div_34, permute_1210, permute_1211, permute_1217, permute_1218, alias_29, permute_1224, permute_1225, permute_1231, permute_1232, permute_1242, permute_1243, permute_1250, permute_1257, div_35, permute_1262, permute_1266, div_36, permute_1271, permute_1272, permute_1278, permute_1279, alias_30, permute_1285, permute_1286, permute_1292, permute_1293, permute_1303, permute_1304, permute_1311, permute_1318, div_37, permute_1323, permute_1327, div_38, permute_1332, permute_1333, permute_1339, permute_1340, alias_31, permute_1346, permute_1347, permute_1353, permute_1354, permute_1364, permute_1365, permute_1372, permute_1379, div_39, permute_1384, permute_1388, div_40, permute_1393, permute_1394, permute_1400, permute_1401, alias_32, permute_1407, permute_1408, permute_1414, permute_1415, permute_1425, permute_1426, permute_1433, permute_1440, div_41, permute_1445, permute_1449, div_42, permute_1454, permute_1455, permute_1461, permute_1462, alias_33, permute_1468, permute_1469, permute_1475, permute_1476, permute_1486, permute_1487, permute_1494, permute_1501, div_43, permute_1506, permute_1510, div_44, permute_1515, permute_1516, permute_1522, permute_1523, alias_34, permute_1529, permute_1530, permute_1536, permute_1537, permute_1547, permute_1548, permute_1555, permute_1562, div_45, permute_1567, permute_1571, div_46, permute_1576, permute_1577, permute_1583, permute_1584, alias_35, permute_1590, permute_1591, permute_1597, permute_1598, permute_1608, permute_1609, permute_1616, permute_1623, div_47, permute_1628, permute_1632, div_48, permute_1637, permute_1638, permute_1644, permute_1645, alias_36, permute_1651, permute_1652, permute_1658, permute_1659, permute_1669, permute_1670, permute_1677, permute_1684, div_49, permute_1689, permute_1693, div_50, permute_1698, permute_1699, permute_1705, permute_1706, alias_37, permute_1712, permute_1713, permute_1719, permute_1720, permute_1730, permute_1731, permute_1738, permute_1745, div_51, permute_1750, permute_1754, div_52, permute_1759, permute_1760, permute_1766, permute_1767, alias_38, permute_1773, permute_1774, permute_1780, permute_1781, permute_1791, permute_1792, permute_1799, permute_1806, div_53, permute_1811, permute_1815, div_54, permute_1820, permute_1821, permute_1827, permute_1828, alias_39, permute_1834, permute_1835, permute_1841, permute_1842, permute_1852, permute_1853, permute_1860, permute_1867, div_55, permute_1872, permute_1876, div_56, permute_1881, permute_1882, permute_1888, permute_1889, alias_40, permute_1895, permute_1896, permute_1902, permute_1903, permute_1913, permute_1914, permute_1921, permute_1928, div_57, permute_1933, permute_1937, div_58, permute_1942, permute_1943, permute_1949, permute_1950, alias_41, permute_1956, permute_1957, permute_1963, permute_1964, permute_1974, permute_1975, permute_1982, permute_1989, div_59, permute_1994, permute_1998, div_60, permute_2003, permute_2004, permute_2010, permute_2011, alias_42, permute_2017, permute_2018, permute_2024, permute_2025, permute_2035, permute_2036, permute_2043, permute_2050, div_61, permute_2055, permute_2059, div_62, permute_2064, permute_2065, permute_2071, permute_2072, alias_43, permute_2078, permute_2079, permute_2085, permute_2086, permute_2096, permute_2097, permute_2104, permute_2111, div_63, permute_2116, permute_2120, div_64, permute_2125, permute_2126, permute_2132, permute_2133, alias_44, permute_2139, permute_2140, permute_2146, permute_2147, permute_2157, permute_2158, permute_2165, permute_2172, div_65, permute_2177, permute_2181, div_66, permute_2186, permute_2187, permute_2193, permute_2194, alias_45, permute_2200, permute_2201, permute_2207, permute_2208, permute_2218, permute_2219, permute_2226, permute_2233, div_67, permute_2238, permute_2242, div_68, permute_2247, permute_2248, permute_2254, permute_2255, alias_46, permute_2261, permute_2262, permute_2268, permute_2269, permute_2279, permute_2280, permute_2287, permute_2294, div_69, permute_2299, permute_2303, div_70, permute_2308, permute_2309, permute_2315, permute_2316, alias_47, permute_2322, permute_2323, permute_2329, permute_2330, permute_2340, permute_2341, permute_2348, permute_2355, div_71, permute_2360, permute_2364, div_72, permute_2369, permute_2370, permute_2376, permute_2377, alias_48, permute_2383, permute_2384, permute_2390, permute_2391, permute_2401, permute_2402, permute_2409, permute_2416, div_73, permute_2421, permute_2425, div_74, permute_2430, permute_2431, permute_2437, permute_2438, alias_49, permute_2444, permute_2445, permute_2451, permute_2452, permute_2462, permute_2463, permute_2470, permute_2477, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XLNetLMHeadModel', benchmark_compiled_module)
