
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


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtuml3y6f2igyp4iaybhd7t52vti7nbilhwpzocgidodnrbxzbm.py
# Source Nodes: [hidden_states_100, hidden_states_105, hidden_states_109, hidden_states_117, hidden_states_122, hidden_states_126, hidden_states_134, hidden_states_139, hidden_states_143, hidden_states_151, hidden_states_156, hidden_states_160, hidden_states_168, hidden_states_173, hidden_states_177, hidden_states_185, hidden_states_88, hidden_states_92], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_100 => add_46
# hidden_states_105 => add_49
# hidden_states_109 => add_52
# hidden_states_117 => add_54
# hidden_states_122 => add_57
# hidden_states_126 => add_60
# hidden_states_134 => add_62
# hidden_states_139 => add_65
# hidden_states_143 => add_68
# hidden_states_151 => add_70
# hidden_states_156 => add_73
# hidden_states_160 => add_76
# hidden_states_168 => add_78
# hidden_states_173 => add_81
# hidden_states_177 => add_84
# hidden_states_185 => add_86
# hidden_states_88 => add_40
# hidden_states_92 => add_44
triton_per_fused_add_div_mul_pow_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: 'i32', 28: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(27, 28))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr5, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask, other=0.0)
    tmp31 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr17 + (r1 + (512*x0)), rmask, other=0.0)
    tmp36 = tl.load(in_ptr18 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr19 + (r1 + (512*x0)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask, other=0.0)
    tmp47 = tl.load(in_ptr21 + (x0), None, eviction_policy='evict_last')
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
    tmp34 = 0.04419417382415922
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp32 + tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp37 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [RBLOCK])
    tmp45 = tl.where(rmask, tmp43, 0)
    tmp46 = triton_helpers.promote_to_tensor(tl.sum(tmp45, 0))
    tmp48 = tmp37 * tmp47
    tmp49 = -0.5
    tmp50 = tmp46 * tmp49
    tmp51 = tmp47 * tmp47
    tmp52 = tmp51 * tmp47
    tmp53 = tmp50 * tmp52
    tmp54 = 512.0
    tmp55 = tmp53 / tmp54
    tmp56 = 2.0
    tmp57 = tmp41 * tmp56
    tmp58 = tmp55 * tmp57
    tmp59 = tmp48 + tmp58
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp16, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp32, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp59, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/zm/czmvily2bniavynwqhhfx7jn4noafxptok746kffdcb63ujztbxh.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6b/c6bnbrdkgdh36glq7fz72ja3nbhazvmydvmqd7j75qutigeo74nq.py
# Source Nodes: [hidden_states_177], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_177 => add_84
triton_per_fused_add_div_mul_pow_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp2 * tmp12
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
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp25, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cflc67inzawjz4444pnw4rooowlpsrg5awv6j4eijw6k4alhtce2.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 8
    x3 = (xindex // 524288)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (524288*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czeg3jnwm2wkmphxjjko6bpjv6wwainrnpne2dpwaekt73vejpq2.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cweuctw6qbr2g45jno3gugupxbenrthuyclm753o7mwruulz6scw.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]

triton_per_fused__softmax_backward_data_as_strided_scatter_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_as_strided_scatter_5', 'mutated_arg_names': ['out_ptr2']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 32768
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
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp8, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp8, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/cei6de4meqgfccntqgiusdlglir3hoh6xd4yqecik4zspdyiwqzf.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_6', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bt5dpo22f7tzhcmkrdxnfukjc3a3t66jv6pr6ylswawqiojhzw.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 1024)) + (65536*(x0 // 64)) + (524288*(x1 // 1024)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4nnsxqk5v4snivmxczkgrga2nxd6gx6ra5dynklnmhcr2nfg6g.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]

triton_per_fused_add_div_mul_pow_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
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
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp23, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6rgulct2odigwujurpqj2ataic6hkr2txnfpfkvhp3xlniwykc.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 8
    x3 = (xindex // 524288)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (512*x1) + (524288*x3)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctou2xduueo3hlo7kyhx634w76rqyfr4w3dpnunbnph7egkkmqqi.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = yindex % 1024
    y5 = (yindex // 1024)
    y1 = (yindex // 1024) % 8
    y2 = (yindex // 8192)
    tmp0 = tl.load(in_ptr0 + (x3 + (64*y4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (1024*x3) + (65536*y5)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3 + (64*y1) + (512*y0) + (524288*y2)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsgxpz3lynahboodwloxor7wac4vvie2zy25z35vphku7su74w2.py
# Source Nodes: [hidden_states_156, hidden_states_160, hidden_states_168], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_156 => add_73
# hidden_states_160 => add_76
# hidden_states_168 => add_78
triton_per_fused_add_div_mul_pow_sum_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp21 = tmp6 * tmp20
    tmp22 = tmp19 + tmp21
    tmp23 = -0.5
    tmp24 = tmp18 * tmp23
    tmp25 = tmp20 * tmp20
    tmp26 = tmp25 * tmp20
    tmp27 = tmp24 * tmp26
    tmp28 = 512.0
    tmp29 = tmp27 / tmp28
    tmp30 = 2.0
    tmp31 = tmp13 * tmp30
    tmp32 = tmp29 * tmp31
    tmp33 = tmp22 + tmp32
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp33, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qj/cqj3ooljl3hycr2qeumbie5g43acpcstq3bjgmfjsdlsscrpn5qh.py
# Source Nodes: [hidden_states_156, hidden_states_160], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_156 => add_73
# hidden_states_160 => add_76
triton_per_fused_add_div_mul_pow_sum_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp2 * tmp14
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
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp27, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qo/cqougvlaykh5jc2xutqgkot3n6utej7xc5aoapbtbcwe7oasas4n.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]

triton_per_fused_add_div_mul_pow_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
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
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp27, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5x5md6nce2timyug3e2ezeiexqpw47l2ludcmxle3dg7t2bfc6u.py
# Source Nodes: [hidden_states_134, hidden_states_139, hidden_states_143], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_134 => add_62
# hidden_states_139 => add_65
# hidden_states_143 => add_68
triton_per_fused_add_div_mul_pow_sum_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp2 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = -0.5
    tmp20 = tmp14 * tmp19
    tmp21 = tmp16 * tmp16
    tmp22 = tmp21 * tmp16
    tmp23 = tmp20 * tmp22
    tmp24 = 512.0
    tmp25 = tmp23 / tmp24
    tmp26 = 2.0
    tmp27 = tmp9 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp18 + tmp28
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp29, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5zkki6levobyrdf77rqjhu7hidfplmshh5t6uupw7iesdjc43r.py
# Source Nodes: [hidden_states_134], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_134 => add_62
triton_per_fused_add_div_mul_pow_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp6 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = -0.5
    tmp20 = tmp14 * tmp19
    tmp21 = tmp16 * tmp16
    tmp22 = tmp21 * tmp16
    tmp23 = tmp20 * tmp22
    tmp24 = 512.0
    tmp25 = tmp23 / tmp24
    tmp26 = 2.0
    tmp27 = tmp9 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp18 + tmp28
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp29, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2l6hbeuwa57tqtfz3biohjhh5w7uqqr23gitsk7unu57l7gywt7.py
# Source Nodes: [hidden_states_109, hidden_states_117], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_109 => add_52
# hidden_states_117 => add_54
triton_per_fused_add_div_mul_pow_sum_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = tmp6 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = -0.5
    tmp22 = tmp16 * tmp21
    tmp23 = tmp18 * tmp18
    tmp24 = tmp23 * tmp18
    tmp25 = tmp22 * tmp24
    tmp26 = 512.0
    tmp27 = tmp25 / tmp26
    tmp28 = 2.0
    tmp29 = tmp11 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp20 + tmp30
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqb3q5ffa6rylv4pfw7bjafmanmlzpcwzf4aqfb3am7qcsyd6it3.py
# Source Nodes: [hidden_states_13, hidden_states_18, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_5, hidden_states_52, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_78], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_13 => add_8
# hidden_states_18 => add_11
# hidden_states_26 => add_13
# hidden_states_31 => add_16
# hidden_states_39 => add_18
# hidden_states_44 => add_21
# hidden_states_5 => add_6
# hidden_states_52 => add_23
# hidden_states_57 => add_26
# hidden_states_65 => add_28
# hidden_states_70 => add_31
# hidden_states_78 => add_33
triton_per_fused_add_div_mul_pow_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: 'i32', 33: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(32, 33))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, out_ptr0, out_ptr1, out_ptr2, out_ptr4, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask, other=0.0)
    tmp26 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask, other=0.0)
    tmp30 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp32 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask, other=0.0)
    tmp34 = tl.load(in_ptr17 + (r1 + (512*x0)), rmask, other=0.0)
    tmp36 = tl.load(in_ptr18 + (r1 + (512*x0)), rmask, other=0.0)
    tmp38 = tl.load(in_ptr19 + (r1 + (512*x0)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask, other=0.0)
    tmp42 = tl.load(in_ptr21 + (r1 + (512*x0)), rmask, other=0.0)
    tmp44 = tl.load(in_ptr22 + (r1 + (512*x0)), rmask, other=0.0)
    tmp46 = tl.load(in_ptr23 + (r1 + (512*x0)), rmask, other=0.0)
    tmp48 = tl.load(in_ptr24 + (r1 + (512*x0)), rmask, other=0.0)
    tmp50 = tl.load(in_ptr25 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.load(in_ptr26 + (x0), None, eviction_policy='evict_last')
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
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 + tmp38
    tmp41 = tmp39 + tmp40
    tmp43 = tmp41 + tmp42
    tmp45 = tmp43 + tmp44
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 * tmp50
    tmp52 = tmp51 * tmp24
    tmp53 = tl.broadcast_to(tmp52, [RBLOCK])
    tmp55 = tl.where(rmask, tmp53, 0)
    tmp56 = triton_helpers.promote_to_tensor(tl.sum(tmp55, 0))
    tmp58 = tmp51 * tmp57
    tmp59 = -0.5
    tmp60 = tmp56 * tmp59
    tmp61 = tmp57 * tmp57
    tmp62 = tmp61 * tmp57
    tmp63 = tmp60 * tmp62
    tmp64 = 512.0
    tmp65 = tmp63 / tmp64
    tmp66 = 2.0
    tmp67 = tmp24 * tmp66
    tmp68 = tmp65 * tmp67
    tmp69 = tmp58 + tmp68
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp16, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp49, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp69, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbgs7ad2qmcvm43mug22st7d65cpage46iki5enr6z773b2iqwpf.py
# Source Nodes: [hidden_states_174, hidden_states_177, hidden_states_178, hidden_states_185, hidden_states_186], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_174 => mul_65
# hidden_states_177 => add_84
# hidden_states_178 => mul_67
# hidden_states_185 => add_86
# hidden_states_186 => mul_69
triton_red_fused_add_mul_sum_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr7 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr8 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.04419417382415922
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp5 * tmp15
        tmp17 = tmp14 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
        tmp23 = tmp3 * tmp22
        tmp24 = tmp21 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chsd5fbkhjdmgijeksjv44oh5duyhcdqbjzcl7ntzhmjpf7eqty7.py
# Source Nodes: [hidden_states_177, hidden_states_185, hidden_states_186], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_177 => add_84
# hidden_states_185 => add_86
# hidden_states_186 => mul_69
triton_per_fused_add_mul_sum_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/im/cimzku42pqpml3kxy5lhrx7dzr56jtusps4pauz37xwshufgjxf5.py
# Source Nodes: [hidden_states_152, hidden_states_156, hidden_states_157, hidden_states_160, hidden_states_161, hidden_states_168, hidden_states_169], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_152 => mul_57
# hidden_states_156 => add_73
# hidden_states_157 => mul_59
# hidden_states_160 => add_76
# hidden_states_161 => mul_61
# hidden_states_168 => add_78
# hidden_states_169 => mul_63
triton_red_fused_add_mul_sum_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20, 21))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr5 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr12 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr13 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr14 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr15 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp4 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp20 = tmp9 * tmp19
        tmp21 = tmp18 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp27 = tmp7 * tmp26
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp5 * tmp37
        tmp39 = tmp36 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask, tmp42, _tmp41)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, None)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp30, None)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp41, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5av7a2bmor67qvp6l4xgy3kakttdzkvkpkqgz5fdxqbjl5foib5.py
# Source Nodes: [hidden_states_127, hidden_states_134, hidden_states_135, hidden_states_139, hidden_states_140, hidden_states_143, hidden_states_144], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_127 => mul_49
# hidden_states_134 => add_62
# hidden_states_135 => mul_51
# hidden_states_139 => add_65
# hidden_states_140 => mul_53
# hidden_states_143 => add_68
# hidden_states_144 => mul_55
triton_red_fused_add_mul_sum_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(18, 19))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp37 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr9 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr12 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr13 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp5 * tmp15
        tmp17 = tmp14 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
        tmp23 = tmp21 + tmp22
        tmp25 = tmp23 + tmp24
        tmp27 = tmp3 * tmp26
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp34 = tmp1 * tmp33
        tmp35 = tmp32 * tmp34
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp38 = _tmp37 + tmp36
        _tmp37 = tl.where(rmask, tmp38, _tmp37)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp30, None)
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdnxv5dl2tyw73o5pul63cfwo3dzgmxpogjzpbkiccbnchtsnv3.py
# Source Nodes: [hidden_states_106, hidden_states_109, hidden_states_110, hidden_states_117, hidden_states_118, hidden_states_122, hidden_states_123], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_106 => mul_41
# hidden_states_109 => add_52
# hidden_states_110 => mul_43
# hidden_states_117 => add_54
# hidden_states_118 => mul_45
# hidden_states_122 => add_57
# hidden_states_123 => mul_47
triton_red_fused_add_mul_sum_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(18, 19))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp37 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr7 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr12 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr13 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp20 = tmp5 * tmp19
        tmp21 = tmp18 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp27 = tmp3 * tmp26
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp34 = tmp1 * tmp33
        tmp35 = tmp32 * tmp34
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp38 = _tmp37 + tmp36
        _tmp37 = tl.where(rmask, tmp38, _tmp37)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp30, None)
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jlsmrrgjrnw5aip4ct72xd7mokculicxoerrbbpngv4442x2cj.py
# Source Nodes: [hidden_states_100, hidden_states_101, hidden_states_88, hidden_states_89, hidden_states_92, hidden_states_93], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_100 => add_46
# hidden_states_101 => mul_39
# hidden_states_88 => add_40
# hidden_states_89 => mul_35
# hidden_states_92 => add_44
# hidden_states_93 => mul_37
triton_red_fused_add_mul_sum_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr5 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp4 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp20 = tmp9 * tmp19
        tmp21 = tmp18 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp27 = tmp7 * tmp26
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, None)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4tgmnodlq24ndft6ehaa7ixa67zt7mehu6n2bycxd7zjyojk5l.py
# Source Nodes: [hidden_states_88], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_88 => add_40
triton_per_fused_add_div_mul_pow_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp2 * tmp12
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
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp25, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qataqppfxcpd2e5p4ozqhelatymfnruskua242lyboa3fxiq4d.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.sum, aten.threshold_backward]

triton_poi_fused_add_embedding_dense_backward_sum_threshold_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_sum_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp9 = tl.load(in_ptr5 + (x0), None)
    tmp11 = tl.load(in_ptr0 + (8388608 + x0), None)
    tmp12 = tl.load(in_ptr1 + (8388608 + x0), None)
    tmp14 = tl.load(in_ptr2 + (8388608 + x0), None)
    tmp16 = tl.load(in_ptr3 + (8388608 + x0), None)
    tmp18 = tl.load(in_ptr4 + (8388608 + x0), None)
    tmp20 = tl.load(in_ptr5 + (8388608 + x0), None)
    tmp23 = tl.load(in_ptr0 + (16777216 + x0), None)
    tmp24 = tl.load(in_ptr1 + (16777216 + x0), None)
    tmp26 = tl.load(in_ptr2 + (16777216 + x0), None)
    tmp28 = tl.load(in_ptr3 + (16777216 + x0), None)
    tmp30 = tl.load(in_ptr4 + (16777216 + x0), None)
    tmp32 = tl.load(in_ptr5 + (16777216 + x0), None)
    tmp35 = tl.load(in_ptr0 + (25165824 + x0), None)
    tmp36 = tl.load(in_ptr1 + (25165824 + x0), None)
    tmp38 = tl.load(in_ptr2 + (25165824 + x0), None)
    tmp40 = tl.load(in_ptr3 + (25165824 + x0), None)
    tmp42 = tl.load(in_ptr4 + (25165824 + x0), None)
    tmp44 = tl.load(in_ptr5 + (25165824 + x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp22 + tmp33
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 + tmp38
    tmp41 = tmp39 + tmp40
    tmp43 = tmp41 + tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp34 + tmp45
    tmp47 = tl.full([1], False, tl.int1)
    tmp48 = 0.0
    tmp49 = tl.where(tmp47, tmp48, tmp46)
    tl.store(in_out_ptr0 + (x0), tmp49, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7jf73nqzog73bhx7dwf2j2p5fyggf2yv5zlrfbjygp5bxpg2rl.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.threshold_backward]

triton_poi_fused_embedding_dense_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_threshold_backward_26', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pb32owcazdhsa76lb4rl3ie5d6hu6gpdtuvs5zb3a2hahaydus.py
# Source Nodes: [hidden_states_84], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_84 => mul_32
triton_red_fused_add_mul_sum_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ng/cngmclinndj6orv7ayu3kvyichrvohr343b632jujjn5o3rlkwyd.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.pow, aten.sum, aten.threshold_backward]

triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.full([1], -1, tl.int64)
    tmp15 = tmp13 == tmp14
    tmp18 = tmp6 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = -0.5
    tmp21 = tmp12 * tmp20
    tmp22 = tmp17 * tmp17
    tmp23 = tmp22 * tmp17
    tmp24 = tmp21 * tmp23
    tmp25 = 512.0
    tmp26 = tmp24 / tmp25
    tmp27 = 2.0
    tmp28 = tmp7 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp19 + tmp29
    tmp31 = 0.0
    tmp32 = tl.where(tmp15, tmp31, tmp30)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp32, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cni65ooazwtasmkeupgy2q5phax5nyjotpa3ej4rlxzwnlxpy7bt.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_29', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zw/czwwyweszreotc7423gmy3f6ypqdkhdkixukqunkvmpbfkkziwim.py
# Source Nodes: [hidden_states_79], Original ATen: [aten.mul, aten.sum]
# hidden_states_79 => mul_27
triton_red_fused_mul_sum_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gwaopl2uvpwbxdy4xlmfcfsnjssx6g646qxhc2hkrjihxx4lww.py
# Source Nodes: [hidden_states_57, hidden_states_65, hidden_states_70], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# hidden_states_57 => add_26
# hidden_states_65 => add_28
# hidden_states_70 => add_31
triton_per_fused_add_div_mul_pow_sum_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_31', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp2 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = -0.5
    tmp20 = tmp14 * tmp19
    tmp21 = tmp16 * tmp16
    tmp22 = tmp21 * tmp16
    tmp23 = tmp20 * tmp22
    tmp24 = 512.0
    tmp25 = tmp23 / tmp24
    tmp26 = 2.0
    tmp27 = tmp9 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp18 + tmp28
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp29, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/ceh74l2ykx5lz2czommvf2bcw2cj6e2h3cp5g5fntxuw733ijpfc.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((1024*x1) + (524288*(y0 // 1024)) + (y0 % 1024)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsxhtfayfelkfo3ohn2pa5bblzpldoxz5asz33wcbsf54c4wvsr.py
# Source Nodes: [hidden_states_53, hidden_states_57, hidden_states_58, hidden_states_65, hidden_states_66, hidden_states_70, hidden_states_71], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_53 => mul_19
# hidden_states_57 => add_26
# hidden_states_58 => mul_21
# hidden_states_65 => add_28
# hidden_states_66 => mul_23
# hidden_states_70 => add_31
# hidden_states_71 => mul_25
triton_red_fused_add_mul_sum_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20, 21))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr7 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr12 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr13 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr14 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr15 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp20 = tmp5 * tmp19
        tmp21 = tmp18 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp27 = tmp3 * tmp26
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp1 * tmp37
        tmp39 = tmp36 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask, tmp42, _tmp41)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp30, None)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp41, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpc6ov4ft5p4gtqhbnnbd7ij5pjivbjd5xz64sggjkwlslbxjefv.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]

triton_per_fused_add_div_mul_pow_sum_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
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
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp27, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnifvws62kir5kjhtbrqrtvdq3y64k56qss3w7kp63baccpcz7k.py
# Source Nodes: [hidden_states_13, hidden_states_14, hidden_states_18, hidden_states_19, hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.mul, aten.sum]
# hidden_states_13 => add_8
# hidden_states_14 => mul_7
# hidden_states_18 => add_11
# hidden_states_19 => mul_9
# hidden_states_5 => add_6
# hidden_states_6 => mul_5
triton_red_fused_add_mul_sum_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr7 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp20 = tmp5 * tmp19
        tmp21 = tmp18 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp27 = tmp3 * tmp26
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cun2nhsj64rmnv2jildm3ykrm3fvgpntbjywioam7tnz5oipmz3s.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_36', 'mutated_arg_names': ['in_out_ptr0']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, view, embedding, rsqrt, view_1, add_3, view_19, mm_3, rsqrt_1, view_21, view_23, mm_5, rsqrt_2, view_25, view_43, mm_9, rsqrt_3, view_45, view_47, mm_11, rsqrt_4, view_49, view_67, mm_15, rsqrt_5, view_69, view_71, mm_17, rsqrt_6, view_73, view_91, mm_21, rsqrt_7, view_93, view_95, mm_23, rsqrt_8, view_97, view_115, mm_27, rsqrt_9, view_117, view_119, mm_29, rsqrt_10, view_121, view_139, mm_33, rsqrt_11, view_141, view_143, mm_35, rsqrt_12, view_145, embedding_2, rsqrt_13, view_146, add_37, view_164, mm_39, rsqrt_14, view_166, view_169, view_184, mm_43, rsqrt_15, view_186, view_188, mm_45, rsqrt_16, view_190, view_208, mm_49, rsqrt_17, view_210, view_228, mm_53, rsqrt_18, view_230, view_232, mm_55, rsqrt_19, view_234, view_252, mm_59, rsqrt_20, view_254, view_272, mm_63, rsqrt_21, view_274, view_276, mm_65, rsqrt_22, view_278, view_296, mm_69, rsqrt_23, view_298, view_316, mm_73, rsqrt_24, view_318, view_320, mm_75, rsqrt_25, view_322, view_340, mm_79, rsqrt_26, view_342, view_360, mm_83, rsqrt_27, view_362, view_364, mm_85, rsqrt_28, view_366, view_384, mm_89, rsqrt_29, view_386, view_404, mm_93, rsqrt_30, view_406, view_408, mm_95, rsqrt_31, view_410, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_65, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_67, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_71, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_73, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_77, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_79, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_83, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_85, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_89, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_91, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_95, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_97, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_102, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_106, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_110, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_114, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_118, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_122, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26 = args
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
    assert_size_stride(view, (4, 1024), (1024, 1))
    assert_size_stride(embedding, (4, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_1, (4096, 512), (512, 1))
    assert_size_stride(add_3, (1024, 1024), (1024, 1))
    assert_size_stride(view_19, (4096, 512), (512, 1))
    assert_size_stride(mm_3, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_1, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_21, (4096, 512), (512, 1))
    assert_size_stride(view_23, (4096, 2048), (2048, 1))
    assert_size_stride(mm_5, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_2, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_25, (4096, 512), (512, 1))
    assert_size_stride(view_43, (4096, 512), (512, 1))
    assert_size_stride(mm_9, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_3, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_45, (4096, 512), (512, 1))
    assert_size_stride(view_47, (4096, 2048), (2048, 1))
    assert_size_stride(mm_11, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_4, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_49, (4096, 512), (512, 1))
    assert_size_stride(view_67, (4096, 512), (512, 1))
    assert_size_stride(mm_15, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_5, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_69, (4096, 512), (512, 1))
    assert_size_stride(view_71, (4096, 2048), (2048, 1))
    assert_size_stride(mm_17, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_6, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_73, (4096, 512), (512, 1))
    assert_size_stride(view_91, (4096, 512), (512, 1))
    assert_size_stride(mm_21, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_7, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_93, (4096, 512), (512, 1))
    assert_size_stride(view_95, (4096, 2048), (2048, 1))
    assert_size_stride(mm_23, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_8, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_97, (4096, 512), (512, 1))
    assert_size_stride(view_115, (4096, 512), (512, 1))
    assert_size_stride(mm_27, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_9, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_117, (4096, 512), (512, 1))
    assert_size_stride(view_119, (4096, 2048), (2048, 1))
    assert_size_stride(mm_29, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_10, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_121, (4096, 512), (512, 1))
    assert_size_stride(view_139, (4096, 512), (512, 1))
    assert_size_stride(mm_33, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_11, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_141, (4096, 512), (512, 1))
    assert_size_stride(view_143, (4096, 2048), (2048, 1))
    assert_size_stride(mm_35, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_12, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_145, (4, 1024), (1024, 1))
    assert_size_stride(embedding_2, (4, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_13, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_146, (4096, 512), (512, 1))
    assert_size_stride(add_37, (1024, 1024), (1024, 1))
    assert_size_stride(view_164, (4096, 512), (512, 1))
    assert_size_stride(mm_39, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_14, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_166, (4096, 512), (512, 1))
    assert_size_stride(view_169, (4096, 512), (512, 1))
    assert_size_stride(view_184, (4096, 512), (512, 1))
    assert_size_stride(mm_43, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_15, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_186, (4096, 512), (512, 1))
    assert_size_stride(view_188, (4096, 2048), (2048, 1))
    assert_size_stride(mm_45, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_16, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_190, (4096, 512), (512, 1))
    assert_size_stride(view_208, (4096, 512), (512, 1))
    assert_size_stride(mm_49, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_17, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_210, (4096, 512), (512, 1))
    assert_size_stride(view_228, (4096, 512), (512, 1))
    assert_size_stride(mm_53, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_18, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_230, (4096, 512), (512, 1))
    assert_size_stride(view_232, (4096, 2048), (2048, 1))
    assert_size_stride(mm_55, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_19, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_234, (4096, 512), (512, 1))
    assert_size_stride(view_252, (4096, 512), (512, 1))
    assert_size_stride(mm_59, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_20, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_254, (4096, 512), (512, 1))
    assert_size_stride(view_272, (4096, 512), (512, 1))
    assert_size_stride(mm_63, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_21, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_274, (4096, 512), (512, 1))
    assert_size_stride(view_276, (4096, 2048), (2048, 1))
    assert_size_stride(mm_65, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_22, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_278, (4096, 512), (512, 1))
    assert_size_stride(view_296, (4096, 512), (512, 1))
    assert_size_stride(mm_69, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_23, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_298, (4096, 512), (512, 1))
    assert_size_stride(view_316, (4096, 512), (512, 1))
    assert_size_stride(mm_73, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_24, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_318, (4096, 512), (512, 1))
    assert_size_stride(view_320, (4096, 2048), (2048, 1))
    assert_size_stride(mm_75, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_25, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_322, (4096, 512), (512, 1))
    assert_size_stride(view_340, (4096, 512), (512, 1))
    assert_size_stride(mm_79, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_26, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_342, (4096, 512), (512, 1))
    assert_size_stride(view_360, (4096, 512), (512, 1))
    assert_size_stride(mm_83, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_27, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_362, (4096, 512), (512, 1))
    assert_size_stride(view_364, (4096, 2048), (2048, 1))
    assert_size_stride(mm_85, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_28, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_366, (4096, 512), (512, 1))
    assert_size_stride(view_384, (4096, 512), (512, 1))
    assert_size_stride(mm_89, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_29, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_386, (4096, 512), (512, 1))
    assert_size_stride(view_404, (4096, 512), (512, 1))
    assert_size_stride(mm_93, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_30, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_406, (4096, 512), (512, 1))
    assert_size_stride(view_408, (4096, 2048), (2048, 1))
    assert_size_stride(mm_95, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_31, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_410, (4096, 512), (512, 1))
    assert_size_stride(permute_191, (32128, 512), (512, 1))
    assert_size_stride(permute_195, (512, 2048), (2048, 1))
    assert_size_stride(le_1, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_199, (2048, 512), (512, 1))
    assert_size_stride(permute_203, (512, 512), (512, 1))
    assert_size_stride(permute_206, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_207, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_65, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_208, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_209, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_214, (512, 512), (512, 1))
    assert_size_stride(permute_219, (512, 512), (512, 1))
    assert_size_stride(permute_224, (512, 512), (512, 1))
    assert_size_stride(permute_228, (512, 512), (512, 1))
    assert_size_stride(permute_231, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_232, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_67, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_233, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_234, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_239, (512, 512), (512, 1))
    assert_size_stride(permute_244, (512, 512), (512, 1))
    assert_size_stride(permute_249, (512, 512), (512, 1))
    assert_size_stride(permute_253, (512, 2048), (2048, 1))
    assert_size_stride(le_2, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_257, (2048, 512), (512, 1))
    assert_size_stride(permute_261, (512, 512), (512, 1))
    assert_size_stride(permute_264, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_265, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_71, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_266, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_267, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_272, (512, 512), (512, 1))
    assert_size_stride(permute_277, (512, 512), (512, 1))
    assert_size_stride(permute_282, (512, 512), (512, 1))
    assert_size_stride(permute_286, (512, 512), (512, 1))
    assert_size_stride(permute_289, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_290, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_73, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_291, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_292, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_297, (512, 512), (512, 1))
    assert_size_stride(permute_302, (512, 512), (512, 1))
    assert_size_stride(permute_307, (512, 512), (512, 1))
    assert_size_stride(permute_311, (512, 2048), (2048, 1))
    assert_size_stride(le_3, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_315, (2048, 512), (512, 1))
    assert_size_stride(permute_319, (512, 512), (512, 1))
    assert_size_stride(permute_322, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_323, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_77, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_324, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_325, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_330, (512, 512), (512, 1))
    assert_size_stride(permute_335, (512, 512), (512, 1))
    assert_size_stride(permute_340, (512, 512), (512, 1))
    assert_size_stride(permute_344, (512, 512), (512, 1))
    assert_size_stride(permute_347, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_348, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_79, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_349, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_350, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_355, (512, 512), (512, 1))
    assert_size_stride(permute_360, (512, 512), (512, 1))
    assert_size_stride(permute_365, (512, 512), (512, 1))
    assert_size_stride(permute_369, (512, 2048), (2048, 1))
    assert_size_stride(le_4, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_373, (2048, 512), (512, 1))
    assert_size_stride(permute_377, (512, 512), (512, 1))
    assert_size_stride(permute_380, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_381, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_83, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_382, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_383, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_388, (512, 512), (512, 1))
    assert_size_stride(permute_393, (512, 512), (512, 1))
    assert_size_stride(permute_398, (512, 512), (512, 1))
    assert_size_stride(permute_402, (512, 512), (512, 1))
    assert_size_stride(permute_405, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_406, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_85, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_407, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_408, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_413, (512, 512), (512, 1))
    assert_size_stride(permute_418, (512, 512), (512, 1))
    assert_size_stride(permute_423, (512, 512), (512, 1))
    assert_size_stride(permute_427, (512, 2048), (2048, 1))
    assert_size_stride(le_5, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_431, (2048, 512), (512, 1))
    assert_size_stride(permute_435, (512, 512), (512, 1))
    assert_size_stride(permute_438, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_439, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_89, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_440, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_441, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_446, (512, 512), (512, 1))
    assert_size_stride(permute_451, (512, 512), (512, 1))
    assert_size_stride(permute_456, (512, 512), (512, 1))
    assert_size_stride(permute_460, (512, 512), (512, 1))
    assert_size_stride(permute_463, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_464, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_91, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_465, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_466, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_471, (512, 512), (512, 1))
    assert_size_stride(permute_476, (512, 512), (512, 1))
    assert_size_stride(permute_481, (512, 512), (512, 1))
    assert_size_stride(permute_485, (512, 2048), (2048, 1))
    assert_size_stride(le_6, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_489, (2048, 512), (512, 1))
    assert_size_stride(permute_493, (512, 512), (512, 1))
    assert_size_stride(permute_496, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_497, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_95, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_498, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_499, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_504, (512, 512), (512, 1))
    assert_size_stride(permute_509, (512, 512), (512, 1))
    assert_size_stride(permute_514, (512, 512), (512, 1))
    assert_size_stride(permute_518, (512, 512), (512, 1))
    assert_size_stride(permute_521, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_522, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_97, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_524, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_525, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_530, (512, 512), (512, 1))
    assert_size_stride(permute_535, (512, 512), (512, 1))
    assert_size_stride(permute_540, (512, 512), (512, 1))
    assert_size_stride(permute_544, (512, 2048), (2048, 1))
    assert_size_stride(le_7, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_548, (2048, 512), (512, 1))
    assert_size_stride(permute_552, (512, 512), (512, 1))
    assert_size_stride(permute_555, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_556, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_102, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_557, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_558, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_563, (512, 512), (512, 1))
    assert_size_stride(permute_568, (512, 512), (512, 1))
    assert_size_stride(permute_573, (512, 512), (512, 1))
    assert_size_stride(permute_577, (512, 2048), (2048, 1))
    assert_size_stride(le_8, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_581, (2048, 512), (512, 1))
    assert_size_stride(permute_585, (512, 512), (512, 1))
    assert_size_stride(permute_588, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_589, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_106, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_590, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_591, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_596, (512, 512), (512, 1))
    assert_size_stride(permute_601, (512, 512), (512, 1))
    assert_size_stride(permute_606, (512, 512), (512, 1))
    assert_size_stride(permute_610, (512, 2048), (2048, 1))
    assert_size_stride(le_9, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_614, (2048, 512), (512, 1))
    assert_size_stride(permute_618, (512, 512), (512, 1))
    assert_size_stride(permute_621, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_622, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_110, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_623, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_624, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_629, (512, 512), (512, 1))
    assert_size_stride(permute_634, (512, 512), (512, 1))
    assert_size_stride(permute_639, (512, 512), (512, 1))
    assert_size_stride(permute_643, (512, 2048), (2048, 1))
    assert_size_stride(le_10, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_647, (2048, 512), (512, 1))
    assert_size_stride(permute_651, (512, 512), (512, 1))
    assert_size_stride(permute_654, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_655, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_114, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_656, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_657, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_662, (512, 512), (512, 1))
    assert_size_stride(permute_667, (512, 512), (512, 1))
    assert_size_stride(permute_672, (512, 512), (512, 1))
    assert_size_stride(permute_676, (512, 2048), (2048, 1))
    assert_size_stride(le_11, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_680, (2048, 512), (512, 1))
    assert_size_stride(permute_684, (512, 512), (512, 1))
    assert_size_stride(permute_687, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_688, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_118, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_689, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_690, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_695, (512, 512), (512, 1))
    assert_size_stride(permute_700, (512, 512), (512, 1))
    assert_size_stride(permute_705, (512, 512), (512, 1))
    assert_size_stride(permute_709, (512, 2048), (2048, 1))
    assert_size_stride(le_12, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_713, (2048, 512), (512, 1))
    assert_size_stride(permute_717, (512, 512), (512, 1))
    assert_size_stride(permute_720, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_721, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_122, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_723, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_724, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_729, (512, 512), (512, 1))
    assert_size_stride(permute_734, (512, 512), (512, 1))
    assert_size_stride(permute_739, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (4, 1024, 32128), (32899072, 32128, 1))
    assert_size_stride(tangents_2, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_3, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_4, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_5, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_6, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_7, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_8, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_9, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_10, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_11, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_12, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_13, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_14, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_15, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_16, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_17, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_18, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_19, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_20, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_21, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_22, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_23, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_24, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_25, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_26, (4, 1024, 512), (524288, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf8 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (4096, 32128), (32128, 1), 0), permute_191, out=buf8)
        del permute_191
        buf3 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf4 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf5 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf6 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf12 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_100, hidden_states_105, hidden_states_109, hidden_states_117, hidden_states_122, hidden_states_126, hidden_states_134, hidden_states_139, hidden_states_143, hidden_states_151, hidden_states_156, hidden_states_160, hidden_states_168, hidden_states_173, hidden_states_177, hidden_states_185, hidden_states_88, hidden_states_92], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_div_mul_pow_sum_0.run(embedding_2, mm_39, mm_43, mm_45, mm_49, mm_53, mm_55, mm_59, mm_63, mm_65, mm_69, mm_73, mm_75, mm_79, mm_83, mm_85, mm_89, buf8, primals_32, mm_93, mm_95, rsqrt_31, buf3, buf4, buf5, buf6, buf12, 4096, 512, grid=grid(4096), stream=stream0)
        del mm_49
        del mm_63
        del mm_75
        del mm_89
        del primals_32
        buf14 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (4096, 512), (512, 1), 0), permute_195, out=buf14)
        del permute_195
        buf15 = reinterpret_tensor(buf14, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf15, le_1, 8388608, grid=grid(8388608), stream=stream0)
        del le_1
        buf17 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (4096, 2048), (2048, 1), 0), permute_199, out=buf17)
        del permute_199
        buf21 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_177], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_2.run(buf17, primals_31, buf6, mm_93, buf12, rsqrt_30, buf21, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_31
        buf23 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (4096, 512), (512, 1), 0), permute_203, out=buf23)
        del permute_203
        buf24 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf23, buf24, 2097152, grid=grid(2097152), stream=stream0)
        buf26 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (32, 1024, 64), (65536, 64, 1), 0), permute_207, out=buf26)
        del permute_207
        buf30 = empty((33554432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf30, 33554432, grid=grid(33554432), stream=stream0)
        buf33 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf26, alias_65, buf30, buf33, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_65
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf30, buf33, 33554432, grid=grid(33554432), stream=stream0)
        buf36 = reinterpret_tensor(buf23, (32, 1024, 64), (65536, 64, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf33, permute_209, out=buf36)
        del permute_209
        buf43 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf36, buf43, 2097152, grid=grid(2097152), stream=stream0)
        buf45 = reinterpret_tensor(buf36, (4096, 512), (512, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf43, permute_224, out=buf45)
        del permute_224
        buf49 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_8.run(buf45, primals_30, buf6, buf21, rsqrt_29, buf49, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_30
        buf51 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (4096, 512), (512, 1), 0), permute_228, out=buf51)
        del permute_228
        buf52 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf51, buf52, 2097152, grid=grid(2097152), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (32, 1024, 64), (65536, 64, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_231, reinterpret_tensor(buf52, (32, 1024, 64), (65536, 64, 1), 0), out=buf53)
        del permute_231
        buf64 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_23, buf53, buf64, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_23
        buf66 = reinterpret_tensor(buf53, (4096, 512), (512, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (4096, 512), (512, 1), 0), permute_239, out=buf66)
        del permute_239
        buf54 = reinterpret_tensor(buf30, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf52, (32, 1024, 64), (65536, 64, 1), 0), permute_232, out=buf54)
        del permute_232
        buf57 = reinterpret_tensor(buf26, (33554432, ), (1, ), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf57, 33554432, grid=grid(33554432), stream=stream0)
        buf60 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf54, alias_67, buf57, buf60, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_67
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf57, buf60, 33554432, grid=grid(33554432), stream=stream0)
        buf62 = reinterpret_tensor(buf52, (32, 64, 1024), (65536, 1024, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_233, buf60, out=buf62)
        del permute_233
        buf67 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_22, buf62, buf67, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_22
        buf69 = reinterpret_tensor(buf62, (4096, 512), (512, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (4096, 512), (512, 1), 0), permute_244, out=buf69)
        del permute_244
        buf63 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf60, permute_234, out=buf63)
        del permute_234
        buf70 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf63, buf70, 2097152, grid=grid(2097152), stream=stream0)
        buf72 = reinterpret_tensor(buf63, (4096, 512), (512, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, permute_249, out=buf72)
        del permute_249
        buf76 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156, hidden_states_160, hidden_states_168], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_11.run(buf66, buf69, buf72, primals_29, buf5, mm_79, mm_83, mm_85, buf49, rsqrt_28, buf76, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_29
        buf78 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (4096, 512), (512, 1), 0), permute_253, out=buf78)
        del permute_253
        buf79 = reinterpret_tensor(buf78, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf79, le_2, 8388608, grid=grid(8388608), stream=stream0)
        del le_2
        buf81 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (4096, 2048), (2048, 1), 0), permute_257, out=buf81)
        del permute_257
        buf85 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156, hidden_states_160], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_12.run(buf81, primals_28, buf5, mm_79, mm_83, buf76, rsqrt_27, buf85, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_28
        buf87 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (4096, 512), (512, 1), 0), permute_261, out=buf87)
        del permute_261
        buf88 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf87, buf88, 2097152, grid=grid(2097152), stream=stream0)
        buf89 = reinterpret_tensor(buf87, (32, 1024, 64), (65536, 64, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_264, reinterpret_tensor(buf88, (32, 1024, 64), (65536, 64, 1), 0), out=buf89)
        del permute_264
        buf100 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_21, buf89, buf100, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_21
        buf102 = reinterpret_tensor(buf89, (4096, 512), (512, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (4096, 512), (512, 1), 0), permute_272, out=buf102)
        del permute_272
        buf90 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (32, 1024, 64), (65536, 64, 1), 0), permute_265, out=buf90)
        del permute_265
        buf93 = reinterpret_tensor(buf54, (33554432, ), (1, ), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf93, 33554432, grid=grid(33554432), stream=stream0)
        buf96 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf90, alias_71, buf93, buf96, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_71
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf93, buf96, 33554432, grid=grid(33554432), stream=stream0)
        buf98 = reinterpret_tensor(buf88, (32, 64, 1024), (65536, 1024, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_266, buf96, out=buf98)
        del permute_266
        buf103 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_20, buf98, buf103, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_20
        buf105 = reinterpret_tensor(buf98, (4096, 512), (512, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 512), (512, 1), 0), permute_277, out=buf105)
        del permute_277
        buf99 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf96, permute_267, out=buf99)
        del permute_267
        buf106 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf99, buf106, 2097152, grid=grid(2097152), stream=stream0)
        buf108 = reinterpret_tensor(buf99, (4096, 512), (512, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, permute_282, out=buf108)
        del permute_282
        buf112 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_2.run(buf108, primals_27, buf5, mm_79, buf85, rsqrt_26, buf112, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_27
        buf114 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (4096, 512), (512, 1), 0), permute_286, out=buf114)
        del permute_286
        buf115 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf114, buf115, 2097152, grid=grid(2097152), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (32, 1024, 64), (65536, 64, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_289, reinterpret_tensor(buf115, (32, 1024, 64), (65536, 64, 1), 0), out=buf116)
        del permute_289
        buf127 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_19, buf116, buf127, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_19
        buf129 = reinterpret_tensor(buf116, (4096, 512), (512, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (4096, 512), (512, 1), 0), permute_297, out=buf129)
        del permute_297
        buf117 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (32, 1024, 64), (65536, 64, 1), 0), permute_290, out=buf117)
        del permute_290
        buf120 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf120, 33554432, grid=grid(33554432), stream=stream0)
        buf123 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf117, alias_73, buf120, buf123, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_73
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf120, buf123, 33554432, grid=grid(33554432), stream=stream0)
        buf125 = reinterpret_tensor(buf115, (32, 64, 1024), (65536, 1024, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_291, buf123, out=buf125)
        del permute_291
        buf130 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_18, buf125, buf130, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_18
        buf132 = reinterpret_tensor(buf125, (4096, 512), (512, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (4096, 512), (512, 1), 0), permute_302, out=buf132)
        del permute_302
        buf126 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf123, permute_292, out=buf126)
        del permute_292
        buf133 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf126, buf133, 2097152, grid=grid(2097152), stream=stream0)
        buf135 = reinterpret_tensor(buf126, (4096, 512), (512, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf133, permute_307, out=buf135)
        del permute_307
        buf139 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_13.run(buf129, buf132, buf135, primals_26, buf5, buf112, rsqrt_25, buf139, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_26
        buf141 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (4096, 512), (512, 1), 0), permute_311, out=buf141)
        del permute_311
        buf142 = reinterpret_tensor(buf141, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf142, le_3, 8388608, grid=grid(8388608), stream=stream0)
        del le_3
        buf144 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (4096, 2048), (2048, 1), 0), permute_315, out=buf144)
        del permute_315
        buf148 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134, hidden_states_139, hidden_states_143], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_14.run(buf144, primals_25, buf4, mm_65, mm_69, mm_73, buf139, rsqrt_24, buf148, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_25
        buf150 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (4096, 512), (512, 1), 0), permute_319, out=buf150)
        del permute_319
        buf151 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf150, buf151, 2097152, grid=grid(2097152), stream=stream0)
        buf152 = reinterpret_tensor(buf150, (32, 1024, 64), (65536, 64, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_322, reinterpret_tensor(buf151, (32, 1024, 64), (65536, 64, 1), 0), out=buf152)
        del permute_322
        buf163 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_17, buf152, buf163, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_17
        buf165 = reinterpret_tensor(buf152, (4096, 512), (512, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (4096, 512), (512, 1), 0), permute_330, out=buf165)
        del permute_330
        buf153 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (32, 1024, 64), (65536, 64, 1), 0), permute_323, out=buf153)
        del permute_323
        buf156 = reinterpret_tensor(buf117, (33554432, ), (1, ), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf156, 33554432, grid=grid(33554432), stream=stream0)
        buf159 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf153, alias_77, buf156, buf159, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_77
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf156, buf159, 33554432, grid=grid(33554432), stream=stream0)
        buf161 = reinterpret_tensor(buf151, (32, 64, 1024), (65536, 1024, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_324, buf159, out=buf161)
        del permute_324
        buf166 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_16, buf161, buf166, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_16
        buf168 = reinterpret_tensor(buf161, (4096, 512), (512, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (4096, 512), (512, 1), 0), permute_335, out=buf168)
        del permute_335
        buf162 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf159, permute_325, out=buf162)
        del permute_325
        buf169 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf162, buf169, 2097152, grid=grid(2097152), stream=stream0)
        buf171 = reinterpret_tensor(buf162, (4096, 512), (512, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf169, permute_340, out=buf171)
        del permute_340
        buf175 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134, hidden_states_139], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_12.run(buf171, primals_24, buf4, mm_65, mm_69, buf148, rsqrt_23, buf175, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_24
        buf177 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (4096, 512), (512, 1), 0), permute_344, out=buf177)
        del permute_344
        buf178 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf177, buf178, 2097152, grid=grid(2097152), stream=stream0)
        buf179 = reinterpret_tensor(buf177, (32, 1024, 64), (65536, 64, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_347, reinterpret_tensor(buf178, (32, 1024, 64), (65536, 64, 1), 0), out=buf179)
        del permute_347
        buf190 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_15, buf179, buf190, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_15
        buf192 = reinterpret_tensor(buf179, (4096, 512), (512, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (4096, 512), (512, 1), 0), permute_355, out=buf192)
        del permute_355
        buf180 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf178, (32, 1024, 64), (65536, 64, 1), 0), permute_348, out=buf180)
        del permute_348
        buf183 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf183, 33554432, grid=grid(33554432), stream=stream0)
        buf186 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf180, alias_79, buf183, buf186, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_79
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf183, buf186, 33554432, grid=grid(33554432), stream=stream0)
        buf188 = reinterpret_tensor(buf178, (32, 64, 1024), (65536, 1024, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_349, buf186, out=buf188)
        del permute_349
        buf193 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_14, buf188, buf193, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_14
        buf195 = reinterpret_tensor(buf188, (4096, 512), (512, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (4096, 512), (512, 1), 0), permute_360, out=buf195)
        del permute_360
        buf189 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf186, permute_350, out=buf189)
        del permute_350
        buf196 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf189, buf196, 2097152, grid=grid(2097152), stream=stream0)
        buf198 = reinterpret_tensor(buf189, (4096, 512), (512, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf196, permute_365, out=buf198)
        del permute_365
        buf202 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_15.run(buf192, buf195, buf198, primals_23, buf4, mm_65, buf175, rsqrt_22, buf202, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_23
        buf204 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (4096, 512), (512, 1), 0), permute_369, out=buf204)
        del permute_369
        buf205 = reinterpret_tensor(buf204, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf205, le_4, 8388608, grid=grid(8388608), stream=stream0)
        del le_4
        buf207 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (4096, 2048), (2048, 1), 0), permute_373, out=buf207)
        del permute_373
        buf211 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_8.run(buf207, primals_22, buf4, buf202, rsqrt_21, buf211, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_22
        buf213 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (4096, 512), (512, 1), 0), permute_377, out=buf213)
        del permute_377
        buf214 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf213, buf214, 2097152, grid=grid(2097152), stream=stream0)
        buf215 = reinterpret_tensor(buf213, (32, 1024, 64), (65536, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_380, reinterpret_tensor(buf214, (32, 1024, 64), (65536, 64, 1), 0), out=buf215)
        del permute_380
        buf226 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_13, buf215, buf226, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_13
        buf228 = reinterpret_tensor(buf215, (4096, 512), (512, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (4096, 512), (512, 1), 0), permute_388, out=buf228)
        del permute_388
        buf216 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf214, (32, 1024, 64), (65536, 64, 1), 0), permute_381, out=buf216)
        del permute_381
        buf219 = reinterpret_tensor(buf180, (33554432, ), (1, ), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf219, 33554432, grid=grid(33554432), stream=stream0)
        buf222 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf216, alias_83, buf219, buf222, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_83
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf219, buf222, 33554432, grid=grid(33554432), stream=stream0)
        buf224 = reinterpret_tensor(buf214, (32, 64, 1024), (65536, 1024, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_382, buf222, out=buf224)
        del permute_382
        buf229 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_12, buf224, buf229, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_12
        buf231 = reinterpret_tensor(buf224, (4096, 512), (512, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (4096, 512), (512, 1), 0), permute_393, out=buf231)
        del permute_393
        buf225 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf222, permute_383, out=buf225)
        del permute_383
        buf233 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf225, buf233, 2097152, grid=grid(2097152), stream=stream0)
        buf235 = reinterpret_tensor(buf225, (4096, 512), (512, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf233, permute_398, out=buf235)
        del permute_398
        buf239 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_117, hidden_states_122], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_14.run(buf235, primals_21, buf3, mm_53, mm_55, mm_59, buf211, rsqrt_20, buf239, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_21
        buf241 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (4096, 512), (512, 1), 0), permute_402, out=buf241)
        del permute_402
        buf242 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf241, buf242, 2097152, grid=grid(2097152), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (32, 1024, 64), (65536, 64, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_405, reinterpret_tensor(buf242, (32, 1024, 64), (65536, 64, 1), 0), out=buf243)
        del permute_405
        buf254 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_11, buf243, buf254, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_11
        buf256 = reinterpret_tensor(buf243, (4096, 512), (512, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (4096, 512), (512, 1), 0), permute_413, out=buf256)
        del permute_413
        buf244 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (32, 1024, 64), (65536, 64, 1), 0), permute_406, out=buf244)
        del permute_406
        buf247 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf247, 33554432, grid=grid(33554432), stream=stream0)
        buf250 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf244, alias_85, buf247, buf250, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_85
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf247, buf250, 33554432, grid=grid(33554432), stream=stream0)
        buf252 = reinterpret_tensor(buf242, (32, 64, 1024), (65536, 1024, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_407, buf250, out=buf252)
        del permute_407
        buf257 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_10, buf252, buf257, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_10
        buf259 = reinterpret_tensor(buf252, (4096, 512), (512, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (4096, 512), (512, 1), 0), permute_418, out=buf259)
        del permute_418
        buf253 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf250, permute_408, out=buf253)
        del permute_408
        buf260 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf253, buf260, 2097152, grid=grid(2097152), stream=stream0)
        buf262 = reinterpret_tensor(buf253, (4096, 512), (512, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf260, permute_423, out=buf262)
        del permute_423
        buf266 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_117], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_16.run(buf256, buf259, buf262, primals_20, buf3, mm_53, mm_55, buf239, rsqrt_19, buf266, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_20
        buf268 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (4096, 512), (512, 1), 0), permute_427, out=buf268)
        del permute_427
        buf269 = reinterpret_tensor(buf268, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf269, le_5, 8388608, grid=grid(8388608), stream=stream0)
        del le_5
        buf271 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (4096, 2048), (2048, 1), 0), permute_431, out=buf271)
        del permute_431
        buf275 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_2.run(buf271, primals_19, buf3, mm_53, buf266, rsqrt_18, buf275, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_19
        buf277 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (4096, 512), (512, 1), 0), permute_435, out=buf277)
        del permute_435
        buf278 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf277, buf278, 2097152, grid=grid(2097152), stream=stream0)
        buf279 = reinterpret_tensor(buf277, (32, 1024, 64), (65536, 64, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_438, reinterpret_tensor(buf278, (32, 1024, 64), (65536, 64, 1), 0), out=buf279)
        del permute_438
        buf290 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_9, buf279, buf290, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_9
        buf292 = reinterpret_tensor(buf279, (4096, 512), (512, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (4096, 512), (512, 1), 0), permute_446, out=buf292)
        del permute_446
        buf280 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (32, 1024, 64), (65536, 64, 1), 0), permute_439, out=buf280)
        del permute_439
        buf283 = reinterpret_tensor(buf244, (33554432, ), (1, ), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf283, 33554432, grid=grid(33554432), stream=stream0)
        buf286 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf280, alias_89, buf283, buf286, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_89
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf283, buf286, 33554432, grid=grid(33554432), stream=stream0)
        buf288 = reinterpret_tensor(buf278, (32, 64, 1024), (65536, 1024, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_440, buf286, out=buf288)
        del permute_440
        buf293 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_8, buf288, buf293, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_8
        buf295 = reinterpret_tensor(buf288, (4096, 512), (512, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (4096, 512), (512, 1), 0), permute_451, out=buf295)
        del permute_451
        buf289 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf286, permute_441, out=buf289)
        del permute_441
        buf296 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf289, buf296, 2097152, grid=grid(2097152), stream=stream0)
        buf298 = reinterpret_tensor(buf289, (4096, 512), (512, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf296, permute_456, out=buf298)
        del permute_456
        buf302 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_8.run(buf298, primals_18, buf3, buf275, rsqrt_17, buf302, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_18
        buf304 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (4096, 512), (512, 1), 0), permute_460, out=buf304)
        del permute_460
        buf305 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf304, buf305, 2097152, grid=grid(2097152), stream=stream0)
        buf306 = reinterpret_tensor(buf304, (32, 1024, 64), (65536, 64, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_463, reinterpret_tensor(buf305, (32, 1024, 64), (65536, 64, 1), 0), out=buf306)
        del permute_463
        buf317 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_7, buf306, buf317, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_7
        buf319 = reinterpret_tensor(buf306, (4096, 512), (512, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (4096, 512), (512, 1), 0), permute_471, out=buf319)
        del permute_471
        buf307 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf305, (32, 1024, 64), (65536, 64, 1), 0), permute_464, out=buf307)
        del permute_464
        buf310 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf310, 33554432, grid=grid(33554432), stream=stream0)
        buf313 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf307, alias_91, buf310, buf313, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_91
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf310, buf313, 33554432, grid=grid(33554432), stream=stream0)
        buf315 = reinterpret_tensor(buf305, (32, 64, 1024), (65536, 1024, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_465, buf313, out=buf315)
        del permute_465
        buf320 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_6, buf315, buf320, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_6
        buf322 = reinterpret_tensor(buf315, (4096, 512), (512, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (4096, 512), (512, 1), 0), permute_476, out=buf322)
        del permute_476
        buf316 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf313, permute_466, out=buf316)
        del permute_466
        buf323 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf316, buf323, 2097152, grid=grid(2097152), stream=stream0)
        buf325 = reinterpret_tensor(buf316, (4096, 512), (512, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf323, permute_481, out=buf325)
        del permute_481
        buf329 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_100, hidden_states_88, hidden_states_92], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_11.run(buf319, buf322, buf325, primals_17, embedding_2, mm_39, mm_43, mm_45, buf302, rsqrt_16, buf329, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_17
        buf331 = empty((4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (4096, 512), (512, 1), 0), permute_485, out=buf331)
        del permute_485
        buf332 = reinterpret_tensor(buf331, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf332, le_6, 8388608, grid=grid(8388608), stream=stream0)
        del le_6
        buf334 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (4096, 2048), (2048, 1), 0), permute_489, out=buf334)
        del permute_489
        buf338 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_88, hidden_states_92], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_12.run(buf334, primals_16, embedding_2, mm_39, mm_43, buf329, rsqrt_15, buf338, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_16
        buf340 = empty((4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (4096, 512), (512, 1), 0), permute_493, out=buf340)
        del permute_493
        buf341 = empty((4, 8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf340, buf341, 2097152, grid=grid(2097152), stream=stream0)
        buf342 = reinterpret_tensor(buf340, (32, 1024, 64), (65536, 64, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_496, reinterpret_tensor(buf341, (32, 1024, 64), (65536, 64, 1), 0), out=buf342)
        del permute_496
        buf353 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_5, buf342, buf353, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_5
        buf355 = reinterpret_tensor(buf342, (4096, 512), (512, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (4096, 512), (512, 1), 0), permute_504, out=buf355)
        del permute_504
        buf343 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf341, (32, 1024, 64), (65536, 64, 1), 0), permute_497, out=buf343)
        del permute_497
        buf346 = reinterpret_tensor(buf307, (33554432, ), (1, ), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf346, 33554432, grid=grid(33554432), stream=stream0)
        buf349 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf343, alias_95, buf346, buf349, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_95
        del buf343
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf346, buf349, 33554432, grid=grid(33554432), stream=stream0)
        buf351 = reinterpret_tensor(buf341, (32, 64, 1024), (65536, 1024, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_498, buf349, out=buf351)
        del permute_498
        buf356 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_4, buf351, buf356, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_4
        buf358 = reinterpret_tensor(buf351, (4096, 512), (512, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (4096, 512), (512, 1), 0), permute_509, out=buf358)
        del permute_509
        buf25 = empty((32, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_206, reinterpret_tensor(buf24, (32, 1024, 64), (65536, 64, 1), 0), out=buf25)
        del permute_206
        buf37 = reinterpret_tensor(buf24, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_25, buf25, buf37, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_25
        buf39 = reinterpret_tensor(buf25, (4096, 512), (512, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (4096, 512), (512, 1), 0), permute_214, out=buf39)
        del permute_214
        buf35 = empty((32, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_208, buf33, out=buf35)
        del permute_208
        buf40 = empty((4, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_24, buf35, buf40, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_24
        buf42 = reinterpret_tensor(buf35, (4096, 512), (512, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (4096, 512), (512, 1), 0), permute_219, out=buf42)
        del permute_219
        buf0 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf1 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf2 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        buf232 = reinterpret_tensor(buf102, (4, 1024, 512), (524288, 512, 1), 0); del buf102  # reuse
        buf359 = buf232; del buf232  # reuse
        buf405 = empty((4, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13, hidden_states_18, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_5, hidden_states_52, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_78], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_17.run(buf359, embedding, mm_3, mm_5, mm_9, mm_11, mm_15, mm_17, mm_21, mm_23, mm_27, mm_29, mm_33, mm_35, tangents_26, buf39, buf42, buf105, buf165, buf168, buf228, buf231, buf292, buf295, buf355, buf358, primals_13, rsqrt_12, buf0, buf1, buf2, buf405, 4096, 512, grid=grid(4096), stream=stream0)
        del buf105
        del buf165
        del buf168
        del buf228
        del buf231
        del buf292
        del buf295
        del buf355
        del buf358
        del buf39
        del buf42
        del mm_11
        del mm_23
        del mm_35
        del primals_13
        del tangents_26
        buf7 = empty((32128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (32128, 4096), (1, 32128), 0), view_410, out=buf7)
        del tangents_1
        del view_410
        buf9 = empty_strided((1, 1, 512, 32), (16384, 16384, 1, 512), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((1, 1, 512, 32), (16384, 16384, 1, 512), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((1, 1, 512, 32), (16384, 16384, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_174, hidden_states_177, hidden_states_178, hidden_states_185, hidden_states_186], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf8, buf6, mm_93, mm_95, rsqrt_31, buf17, rsqrt_30, buf45, rsqrt_29, buf9, buf18, buf46, 16384, 128, grid=grid(16384), stream=stream0)
        del buf17
        del buf45
        del buf6
        del buf8
        del mm_93
        del mm_95
        del rsqrt_29
        del rsqrt_30
        del rsqrt_31
        buf10 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_177, hidden_states_185, hidden_states_186], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf9, buf10, 512, 32, grid=grid(512), stream=stream0)
        buf13 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 4096), (1, 512), 0), view_408, out=buf13)
        del buf12
        del view_408
        buf16 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (2048, 4096), (1, 2048), 0), view_406, out=buf16)
        del buf15
        del view_406
        buf19 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_177, hidden_states_178], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf18, buf19, 512, 32, grid=grid(512), stream=stream0)
        buf22 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 4096), (1, 512), 0), view_404, out=buf22)
        del buf21
        del view_404
        buf38 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (512, 4096), (1, 512), 0), view_169, out=buf38)
        del buf37
        buf41 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (512, 4096), (1, 512), 0), view_169, out=buf41)
        del buf40
        buf44 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (512, 4096), (1, 512), 0), view_386, out=buf44)
        del buf43
        del view_386
        buf47 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_174], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf46, buf47, 512, 32, grid=grid(512), stream=stream0)
        buf50 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (512, 4096), (1, 512), 0), view_384, out=buf50)
        del buf49
        del view_384
        buf65 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 4096), (1, 512), 0), view_366, out=buf65)
        del buf64
        buf68 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (512, 4096), (1, 512), 0), view_366, out=buf68)
        del buf67
        buf71 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 4096), (1, 512), 0), view_366, out=buf71)
        del buf70
        del view_366
        buf73 = buf46; del buf46  # reuse
        buf82 = buf18; del buf18  # reuse
        buf109 = buf9; del buf9  # reuse
        buf136 = empty_strided((1, 1, 512, 32), (16384, 16384, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_152, hidden_states_156, hidden_states_157, hidden_states_160, hidden_states_161, hidden_states_168, hidden_states_169], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf66, buf69, buf72, buf5, mm_79, mm_83, mm_85, rsqrt_28, buf81, rsqrt_27, buf108, rsqrt_26, buf129, buf132, buf135, rsqrt_25, buf73, buf82, buf109, buf136, 16384, 128, grid=grid(16384), stream=stream0)
        del buf108
        del buf129
        del buf132
        del buf135
        del buf5
        del buf66
        del buf69
        del buf72
        del buf81
        del mm_79
        del mm_83
        del mm_85
        del rsqrt_25
        del rsqrt_26
        del rsqrt_27
        del rsqrt_28
        buf74 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156, hidden_states_160, hidden_states_168, hidden_states_169], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf73, buf74, 512, 32, grid=grid(512), stream=stream0)
        buf77 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 4096), (1, 512), 0), view_364, out=buf77)
        del buf76
        del view_364
        buf80 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (2048, 4096), (1, 2048), 0), view_362, out=buf80)
        del buf79
        del view_362
        buf83 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156, hidden_states_160, hidden_states_161], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf82, buf83, 512, 32, grid=grid(512), stream=stream0)
        buf86 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, 4096), (1, 512), 0), view_360, out=buf86)
        del buf85
        del view_360
        buf101 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (512, 4096), (1, 512), 0), view_169, out=buf101)
        del buf100
        buf104 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (512, 4096), (1, 512), 0), view_169, out=buf104)
        del buf103
        buf107 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (512, 4096), (1, 512), 0), view_342, out=buf107)
        del buf106
        del view_342
        buf110 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156, hidden_states_157], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf109, buf110, 512, 32, grid=grid(512), stream=stream0)
        buf113 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (512, 4096), (1, 512), 0), view_340, out=buf113)
        del buf112
        del view_340
        buf128 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (512, 4096), (1, 512), 0), view_322, out=buf128)
        del buf127
        buf131 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (512, 4096), (1, 512), 0), view_322, out=buf131)
        del buf130
        buf134 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (512, 4096), (1, 512), 0), view_322, out=buf134)
        del buf133
        del view_322
        buf137 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_152], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf136, buf137, 512, 32, grid=grid(512), stream=stream0)
        buf140 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (512, 4096), (1, 512), 0), view_320, out=buf140)
        del buf139
        del view_320
        buf143 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (2048, 4096), (1, 2048), 0), view_318, out=buf143)
        del buf142
        del view_318
        buf145 = buf136; del buf136  # reuse
        buf172 = buf109; del buf109  # reuse
        buf199 = buf82; del buf82  # reuse
        buf208 = buf73; del buf73  # reuse
        # Source Nodes: [hidden_states_127, hidden_states_134, hidden_states_135, hidden_states_139, hidden_states_140, hidden_states_143, hidden_states_144], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_21.run(buf144, buf4, mm_65, mm_69, mm_73, rsqrt_24, buf171, rsqrt_23, buf192, buf195, buf198, rsqrt_22, buf207, rsqrt_21, buf145, buf172, buf199, buf208, 16384, 128, grid=grid(16384), stream=stream0)
        del buf144
        del buf171
        del buf192
        del buf195
        del buf198
        del buf207
        del buf4
        del mm_65
        del mm_69
        del mm_73
        del rsqrt_21
        del rsqrt_22
        del rsqrt_23
        del rsqrt_24
        buf146 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134, hidden_states_139, hidden_states_143, hidden_states_144], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf145, buf146, 512, 32, grid=grid(512), stream=stream0)
        buf149 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 4096), (1, 512), 0), view_316, out=buf149)
        del buf148
        del view_316
        buf164 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (512, 4096), (1, 512), 0), view_169, out=buf164)
        del buf163
        buf167 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (1, 512), 0), view_169, out=buf167)
        del buf166
        buf170 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (512, 4096), (1, 512), 0), view_298, out=buf170)
        del buf169
        del view_298
        buf173 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134, hidden_states_139, hidden_states_140], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf172, buf173, 512, 32, grid=grid(512), stream=stream0)
        buf176 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (512, 4096), (1, 512), 0), view_296, out=buf176)
        del buf175
        del view_296
        buf191 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 4096), (1, 512), 0), view_278, out=buf191)
        del buf190
        buf194 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (512, 4096), (1, 512), 0), view_278, out=buf194)
        del buf193
        buf197 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 4096), (1, 512), 0), view_278, out=buf197)
        del buf196
        del view_278
        buf200 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134, hidden_states_135], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf199, buf200, 512, 32, grid=grid(512), stream=stream0)
        buf203 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (512, 4096), (1, 512), 0), view_276, out=buf203)
        del buf202
        del view_276
        buf206 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (2048, 4096), (1, 2048), 0), view_274, out=buf206)
        del buf205
        del view_274
        buf209 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_127], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf208, buf209, 512, 32, grid=grid(512), stream=stream0)
        buf212 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 4096), (1, 512), 0), view_272, out=buf212)
        del buf211
        del view_272
        buf227 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (512, 4096), (1, 512), 0), view_169, out=buf227)
        del buf226
        buf230 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (512, 4096), (1, 512), 0), view_169, out=buf230)
        del buf229
        buf234 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (512, 4096), (1, 512), 0), view_254, out=buf234)
        del buf233
        del view_254
        buf236 = buf208; del buf208  # reuse
        buf263 = buf199; del buf199  # reuse
        buf272 = buf172; del buf172  # reuse
        buf299 = buf145; del buf145  # reuse
        # Source Nodes: [hidden_states_106, hidden_states_109, hidden_states_110, hidden_states_117, hidden_states_118, hidden_states_122, hidden_states_123], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_22.run(buf235, buf3, mm_53, mm_55, mm_59, rsqrt_20, buf256, buf259, buf262, rsqrt_19, buf271, rsqrt_18, buf298, rsqrt_17, buf236, buf263, buf272, buf299, 16384, 128, grid=grid(16384), stream=stream0)
        del buf235
        del buf256
        del buf259
        del buf262
        del buf271
        del buf298
        del buf3
        del mm_53
        del mm_55
        del mm_59
        del rsqrt_17
        del rsqrt_18
        del rsqrt_19
        del rsqrt_20
        buf237 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_117, hidden_states_122, hidden_states_123], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf236, buf237, 512, 32, grid=grid(512), stream=stream0)
        buf240 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (512, 4096), (1, 512), 0), view_252, out=buf240)
        del buf239
        del view_252
        buf255 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (512, 4096), (1, 512), 0), view_234, out=buf255)
        del buf254
        buf258 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (512, 4096), (1, 512), 0), view_234, out=buf258)
        del buf257
        buf261 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 4096), (1, 512), 0), view_234, out=buf261)
        del buf260
        del view_234
        buf264 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_117, hidden_states_118], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf263, buf264, 512, 32, grid=grid(512), stream=stream0)
        buf267 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (512, 4096), (1, 512), 0), view_232, out=buf267)
        del buf266
        del view_232
        buf270 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 4096), (1, 2048), 0), view_230, out=buf270)
        del buf269
        del view_230
        buf273 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_110], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf272, buf273, 512, 32, grid=grid(512), stream=stream0)
        buf276 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (512, 4096), (1, 512), 0), view_228, out=buf276)
        del buf275
        del view_228
        buf291 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (512, 4096), (1, 512), 0), view_169, out=buf291)
        del buf290
        buf294 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (512, 4096), (1, 512), 0), view_169, out=buf294)
        buf297 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (512, 4096), (1, 512), 0), view_210, out=buf297)
        del view_210
        buf300 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_106], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf299, buf300, 512, 32, grid=grid(512), stream=stream0)
        buf303 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (512, 4096), (1, 512), 0), view_208, out=buf303)
        del view_208
        buf318 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (512, 4096), (1, 512), 0), view_190, out=buf318)
        buf321 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (512, 4096), (1, 512), 0), view_190, out=buf321)
        buf324 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (512, 4096), (1, 512), 0), view_190, out=buf324)
        del view_190
        buf352 = reinterpret_tensor(buf323, (32, 1024, 64), (65536, 64, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf349, permute_499, out=buf352)
        del permute_499
        buf360 = reinterpret_tensor(buf320, (4096, 512), (512, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf352, buf360, 2097152, grid=grid(2097152), stream=stream0)
        buf362 = reinterpret_tensor(buf352, (4096, 512), (512, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf360, permute_514, out=buf362)
        del permute_514
        buf326 = buf299; del buf299  # reuse
        buf335 = buf272; del buf272  # reuse
        buf363 = buf263; del buf263  # reuse
        # Source Nodes: [hidden_states_100, hidden_states_101, hidden_states_88, hidden_states_89, hidden_states_92, hidden_states_93], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf319, buf322, buf325, embedding_2, mm_39, mm_43, mm_45, rsqrt_16, buf334, rsqrt_15, buf362, rsqrt_14, buf326, buf335, buf363, 16384, 128, grid=grid(16384), stream=stream0)
        del mm_43
        del mm_45
        del rsqrt_15
        del rsqrt_16
        buf327 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_100, hidden_states_101, hidden_states_88, hidden_states_92], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf326, buf327, 512, 32, grid=grid(512), stream=stream0)
        buf330 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (512, 4096), (1, 512), 0), view_188, out=buf330)
        del view_188
        buf333 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (2048, 4096), (1, 2048), 0), view_186, out=buf333)
        del view_186
        buf336 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_88, hidden_states_92, hidden_states_93], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf335, buf336, 512, 32, grid=grid(512), stream=stream0)
        buf339 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (512, 4096), (1, 512), 0), view_184, out=buf339)
        del view_184
        buf354 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (512, 4096), (1, 512), 0), view_169, out=buf354)
        buf357 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (512, 4096), (1, 512), 0), view_169, out=buf357)
        del view_169
        buf361 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (512, 4096), (1, 512), 0), view_166, out=buf361)
        del view_166
        buf364 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_88, hidden_states_89], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf363, buf364, 512, 32, grid=grid(512), stream=stream0)
        buf366 = buf338; del buf338  # reuse
        # Source Nodes: [hidden_states_88], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_24.run(buf366, buf362, primals_15, embedding_2, mm_39, rsqrt_14, 4096, 512, grid=grid(4096), stream=stream0)
        del mm_39
        del primals_15
        del rsqrt_14
        buf367 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (512, 4096), (1, 512), 0), view_164, out=buf367)
        del view_164
        buf368 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (4096, 512), (512, 1), 0), permute_518, out=buf368)
        del permute_518
        buf369 = reinterpret_tensor(buf360, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf368, buf369, 2097152, grid=grid(2097152), stream=stream0)
        buf370 = reinterpret_tensor(buf368, (32, 1024, 64), (65536, 64, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_521, reinterpret_tensor(buf369, (32, 1024, 64), (65536, 64, 1), 0), out=buf370)
        del permute_521
        buf371 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (32, 1024, 64), (65536, 64, 1), 0), permute_522, out=buf371)
        del permute_522
        buf374 = reinterpret_tensor(buf33, (33554432, ), (1, ), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf374, 33554432, grid=grid(33554432), stream=stream0)
        buf377 = reinterpret_tensor(buf346, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf371, alias_97, buf374, buf377, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_97
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf374, buf377, 33554432, grid=grid(33554432), stream=stream0)
        buf379 = reinterpret_tensor(buf332, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf332  # reuse
        buf381 = reinterpret_tensor(buf379, (1024, 1024, 8), (1024, 1, 1048576), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.sum, aten.threshold_backward]
        triton_poi_fused_add_embedding_dense_backward_sum_threshold_backward_25.run(buf381, buf57, buf120, buf183, buf247, buf310, buf374, 8388608, grid=grid(8388608), stream=stream0)
        buf380 = empty((32, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.threshold_backward]
        triton_poi_fused_embedding_dense_backward_threshold_backward_26.run(buf380, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf380, [add_37], buf381, True)
        del add_37
        buf384 = reinterpret_tensor(buf369, (32, 64, 1024), (65536, 1024, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_524, buf377, out=buf384)
        del permute_524
        buf385 = reinterpret_tensor(buf356, (32, 1024, 64), (65536, 64, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf377, permute_525, out=buf385)
        del permute_525
        buf386 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(tangents_3, buf370, buf386, 2097152, grid=grid(2097152), stream=stream0)
        del tangents_3
        buf387 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (512, 4096), (1, 512), 0), view_146, out=buf387)
        buf388 = reinterpret_tensor(buf370, (4096, 512), (512, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (4096, 512), (512, 1), 0), permute_530, out=buf388)
        del permute_530
        buf389 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_2, buf384, buf389, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del tangents_2
        buf390 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (512, 4096), (1, 512), 0), view_146, out=buf390)
        buf391 = reinterpret_tensor(buf384, (4096, 512), (512, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (4096, 512), (512, 1), 0), permute_535, out=buf391)
        del permute_535
        buf392 = reinterpret_tensor(buf389, (4096, 512), (512, 1), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf385, buf392, 2097152, grid=grid(2097152), stream=stream0)
        buf393 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (512, 4096), (1, 512), 0), view_146, out=buf393)
        del view_146
        buf394 = reinterpret_tensor(buf385, (4096, 512), (512, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf392, permute_540, out=buf394)
        del permute_540
        buf395 = buf363; del buf363  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_27.run(buf388, buf391, buf394, embedding_2, rsqrt_13, buf395, 16384, 128, grid=grid(16384), stream=stream0)
        buf396 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf395, buf396, 512, 32, grid=grid(512), stream=stream0)
        buf398 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.pow, aten.sum, aten.threshold_backward]
        triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_28.run(buf398, buf388, buf391, buf394, primals_14, embedding_2, view_145, rsqrt_13, 4096, 512, grid=grid(4096), stream=stream0)
        del embedding_2
        del primals_14
        del rsqrt_13
        buf399 = empty((32128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_29.run(buf399, 16449536, grid=grid(16449536), stream=stream0)
        aten.index_put_(buf399, [view_145], buf398, True)
        del view_145
        buf402 = buf395; del buf395  # reuse
        # Source Nodes: [hidden_states_79], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf359, buf2, rsqrt_12, buf402, 16384, 128, grid=grid(16384), stream=stream0)
        del rsqrt_12
        buf403 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_79], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf402, buf403, 512, 32, grid=grid(512), stream=stream0)
        buf406 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (512, 4096), (1, 512), 0), view_143, out=buf406)
        del view_143
        buf407 = reinterpret_tensor(buf381, (4096, 2048), (2048, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (4096, 512), (512, 1), 0), permute_544, out=buf407)
        del permute_544
        buf408 = reinterpret_tensor(buf407, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf408, le_7, 8388608, grid=grid(8388608), stream=stream0)
        del le_7
        buf409 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (2048, 4096), (1, 2048), 0), view_141, out=buf409)
        del view_141
        buf410 = reinterpret_tensor(buf359, (4096, 512), (512, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (4096, 2048), (2048, 1), 0), permute_548, out=buf410)
        del permute_548
        buf414 = buf405; del buf405  # reuse
        # Source Nodes: [hidden_states_57, hidden_states_65, hidden_states_70], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_31.run(buf414, buf410, primals_12, buf1, mm_27, mm_29, mm_33, rsqrt_11, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_12
        buf416 = reinterpret_tensor(buf2, (4096, 512), (512, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf414, (4096, 512), (512, 1), 0), permute_552, out=buf416)
        del permute_552
        buf417 = reinterpret_tensor(buf398, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf416, buf417, 2097152, grid=grid(2097152), stream=stream0)
        buf418 = reinterpret_tensor(buf416, (32, 1024, 64), (65536, 64, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_555, reinterpret_tensor(buf417, (32, 1024, 64), (65536, 64, 1), 0), out=buf418)
        del permute_555
        buf429 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf418, buf429, 2097152, grid=grid(2097152), stream=stream0)
        buf431 = reinterpret_tensor(buf418, (4096, 512), (512, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf429, permute_563, out=buf431)
        del permute_563
        buf419 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf417, (32, 1024, 64), (65536, 64, 1), 0), permute_556, out=buf419)
        del permute_556
        buf422 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf422, 33554432, grid=grid(33554432), stream=stream0)
        buf425 = reinterpret_tensor(buf374, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf419, alias_102, buf422, buf425, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_102
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf422, buf425, 33554432, grid=grid(33554432), stream=stream0)
        buf427 = reinterpret_tensor(buf417, (32, 64, 1024), (65536, 1024, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_557, buf425, out=buf427)
        del permute_557
        buf432 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_32.run(buf427, buf432, 4096, 512, grid=grid(4096, 512), stream=stream0)
        buf434 = reinterpret_tensor(buf427, (4096, 512), (512, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf432, permute_568, out=buf434)
        del permute_568
        buf428 = reinterpret_tensor(buf388, (32, 1024, 64), (65536, 64, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf425, permute_558, out=buf428)
        del permute_558
        buf435 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf428, buf435, 2097152, grid=grid(2097152), stream=stream0)
        buf437 = reinterpret_tensor(buf428, (4096, 512), (512, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf435, permute_573, out=buf437)
        del permute_573
        buf441 = buf329; del buf329  # reuse
        # Source Nodes: [hidden_states_57, hidden_states_65], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_16.run(buf431, buf434, buf437, primals_11, buf1, mm_27, mm_29, buf414, rsqrt_10, buf441, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_11
        buf443 = reinterpret_tensor(buf408, (4096, 2048), (2048, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf441, (4096, 512), (512, 1), 0), permute_577, out=buf443)
        del permute_577
        buf444 = reinterpret_tensor(buf443, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf444, le_8, 8388608, grid=grid(8388608), stream=stream0)
        del le_8
        buf446 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (4096, 2048), (2048, 1), 0), permute_581, out=buf446)
        del permute_581
        buf450 = reinterpret_tensor(buf325, (4, 1024, 512), (524288, 512, 1), 0); del buf325  # reuse
        # Source Nodes: [hidden_states_57], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_2.run(buf446, primals_10, buf1, mm_27, buf441, rsqrt_9, buf450, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_10
        buf452 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (4096, 512), (512, 1), 0), permute_585, out=buf452)
        del permute_585
        buf453 = reinterpret_tensor(buf319, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf452, buf453, 2097152, grid=grid(2097152), stream=stream0)
        buf454 = reinterpret_tensor(buf452, (32, 1024, 64), (65536, 64, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_588, reinterpret_tensor(buf453, (32, 1024, 64), (65536, 64, 1), 0), out=buf454)
        del permute_588
        buf465 = reinterpret_tensor(buf317, (4096, 512), (512, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf454, buf465, 2097152, grid=grid(2097152), stream=stream0)
        buf467 = reinterpret_tensor(buf454, (4096, 512), (512, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf465, permute_596, out=buf467)
        del permute_596
        buf455 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf453, (32, 1024, 64), (65536, 64, 1), 0), permute_589, out=buf455)
        del permute_589
        buf458 = reinterpret_tensor(buf419, (33554432, ), (1, ), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf458, 33554432, grid=grid(33554432), stream=stream0)
        buf461 = reinterpret_tensor(buf310, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf455, alias_106, buf458, buf461, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_106
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf458, buf461, 33554432, grid=grid(33554432), stream=stream0)
        buf463 = reinterpret_tensor(buf453, (32, 64, 1024), (65536, 1024, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_590, buf461, out=buf463)
        del permute_590
        buf468 = reinterpret_tensor(buf302, (4096, 512), (512, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_32.run(buf463, buf468, 4096, 512, grid=grid(4096, 512), stream=stream0)
        buf470 = reinterpret_tensor(buf463, (4096, 512), (512, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf468, permute_601, out=buf470)
        del permute_601
        buf464 = reinterpret_tensor(buf296, (32, 1024, 64), (65536, 64, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf461, permute_591, out=buf464)
        del permute_591
        buf471 = reinterpret_tensor(buf293, (4096, 512), (512, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf464, buf471, 2097152, grid=grid(2097152), stream=stream0)
        buf473 = reinterpret_tensor(buf464, (4096, 512), (512, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf471, permute_606, out=buf473)
        del permute_606
        buf411 = buf402; del buf402  # reuse
        buf438 = buf335; del buf335  # reuse
        buf447 = buf326; del buf326  # reuse
        buf474 = buf236; del buf236  # reuse
        # Source Nodes: [hidden_states_53, hidden_states_57, hidden_states_58, hidden_states_65, hidden_states_66, hidden_states_70, hidden_states_71], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_33.run(buf410, buf1, mm_27, mm_29, mm_33, rsqrt_11, buf431, buf434, buf437, rsqrt_10, buf446, rsqrt_9, buf467, buf470, buf473, rsqrt_8, buf411, buf438, buf447, buf474, 16384, 128, grid=grid(16384), stream=stream0)
        del buf410
        del mm_27
        del mm_29
        del mm_33
        del rsqrt_10
        del rsqrt_11
        del rsqrt_9
        buf412 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_71], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf411, buf412, 512, 32, grid=grid(512), stream=stream0)
        buf415 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf414, (512, 4096), (1, 512), 0), view_139, out=buf415)
        del view_139
        buf430 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (512, 4096), (1, 512), 0), view_121, out=buf430)
        buf433 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (512, 4096), (1, 512), 0), view_121, out=buf433)
        buf436 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (512, 4096), (1, 512), 0), view_121, out=buf436)
        del view_121
        buf439 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57, hidden_states_65, hidden_states_66], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf438, buf439, 512, 32, grid=grid(512), stream=stream0)
        buf442 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf441, (512, 4096), (1, 512), 0), view_119, out=buf442)
        del view_119
        buf445 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (2048, 4096), (1, 2048), 0), view_117, out=buf445)
        del view_117
        buf448 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57, hidden_states_58], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf447, buf448, 512, 32, grid=grid(512), stream=stream0)
        buf451 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (512, 4096), (1, 512), 0), view_115, out=buf451)
        del view_115
        buf466 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (512, 4096), (1, 512), 0), view_97, out=buf466)
        buf469 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (512, 4096), (1, 512), 0), view_97, out=buf469)
        buf472 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (512, 4096), (1, 512), 0), view_97, out=buf472)
        del view_97
        buf475 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_53], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf474, buf475, 512, 32, grid=grid(512), stream=stream0)
        buf477 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_34.run(buf477, buf467, buf470, buf473, primals_9, buf1, rsqrt_8, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_9
        del rsqrt_8
        buf478 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (512, 4096), (1, 512), 0), view_95, out=buf478)
        del view_95
        buf479 = reinterpret_tensor(buf444, (4096, 2048), (2048, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (4096, 512), (512, 1), 0), permute_610, out=buf479)
        del permute_610
        buf480 = reinterpret_tensor(buf479, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf480, le_9, 8388608, grid=grid(8388608), stream=stream0)
        del le_9
        buf481 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (2048, 4096), (1, 2048), 0), view_93, out=buf481)
        del view_93
        buf482 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (4096, 2048), (2048, 1), 0), permute_614, out=buf482)
        del permute_614
        buf486 = buf477; del buf477  # reuse
        # Source Nodes: [hidden_states_31, hidden_states_39, hidden_states_44], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_31.run(buf486, buf482, primals_8, buf0, mm_15, mm_17, mm_21, rsqrt_7, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_8
        buf488 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (4096, 512), (512, 1), 0), permute_618, out=buf488)
        del permute_618
        buf489 = reinterpret_tensor(buf467, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf488, buf489, 2097152, grid=grid(2097152), stream=stream0)
        buf490 = reinterpret_tensor(buf488, (32, 1024, 64), (65536, 64, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_621, reinterpret_tensor(buf489, (32, 1024, 64), (65536, 64, 1), 0), out=buf490)
        del permute_621
        buf501 = reinterpret_tensor(buf1, (4096, 512), (512, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf490, buf501, 2097152, grid=grid(2097152), stream=stream0)
        buf503 = reinterpret_tensor(buf490, (4096, 512), (512, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf501, permute_629, out=buf503)
        del permute_629
        buf491 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (32, 1024, 64), (65536, 64, 1), 0), permute_622, out=buf491)
        del permute_622
        buf494 = reinterpret_tensor(buf455, (33554432, ), (1, ), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf494, 33554432, grid=grid(33554432), stream=stream0)
        buf497 = reinterpret_tensor(buf247, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf491, alias_110, buf494, buf497, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_110
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf494, buf497, 33554432, grid=grid(33554432), stream=stream0)
        buf499 = reinterpret_tensor(buf489, (32, 64, 1024), (65536, 1024, 1), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_623, buf497, out=buf499)
        del permute_623
        buf504 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_32.run(buf499, buf504, 4096, 512, grid=grid(4096, 512), stream=stream0)
        buf506 = reinterpret_tensor(buf499, (4096, 512), (512, 1), 0); del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf504, permute_634, out=buf506)
        del permute_634
        buf500 = reinterpret_tensor(buf468, (32, 1024, 64), (65536, 64, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf497, permute_624, out=buf500)
        del permute_624
        buf507 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf500, buf507, 2097152, grid=grid(2097152), stream=stream0)
        buf509 = reinterpret_tensor(buf500, (4096, 512), (512, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf507, permute_639, out=buf509)
        del permute_639
        buf513 = buf441; del buf441  # reuse
        # Source Nodes: [hidden_states_31, hidden_states_39], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_16.run(buf503, buf506, buf509, primals_7, buf0, mm_15, mm_17, buf486, rsqrt_6, buf513, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_7
        buf515 = reinterpret_tensor(buf480, (4096, 2048), (2048, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf513, (4096, 512), (512, 1), 0), permute_643, out=buf515)
        del permute_643
        buf516 = reinterpret_tensor(buf515, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf516, le_10, 8388608, grid=grid(8388608), stream=stream0)
        del le_10
        buf518 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (4096, 2048), (2048, 1), 0), permute_647, out=buf518)
        del permute_647
        buf522 = reinterpret_tensor(buf432, (4, 1024, 512), (524288, 512, 1), 0); del buf432  # reuse
        # Source Nodes: [hidden_states_31], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_2.run(buf518, primals_6, buf0, mm_15, buf513, rsqrt_5, buf522, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_6
        buf524 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf522, (4096, 512), (512, 1), 0), permute_651, out=buf524)
        del permute_651
        buf525 = reinterpret_tensor(buf414, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf524, buf525, 2097152, grid=grid(2097152), stream=stream0)
        buf526 = reinterpret_tensor(buf524, (32, 1024, 64), (65536, 64, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_654, reinterpret_tensor(buf525, (32, 1024, 64), (65536, 64, 1), 0), out=buf526)
        del permute_654
        buf537 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf526, buf537, 2097152, grid=grid(2097152), stream=stream0)
        buf539 = reinterpret_tensor(buf526, (4096, 512), (512, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf537, permute_662, out=buf539)
        del permute_662
        buf527 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf525, (32, 1024, 64), (65536, 64, 1), 0), permute_655, out=buf527)
        del permute_655
        buf530 = reinterpret_tensor(buf491, (33554432, ), (1, ), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf530, 33554432, grid=grid(33554432), stream=stream0)
        buf533 = reinterpret_tensor(buf183, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf527, alias_114, buf530, buf533, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_114
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf530, buf533, 33554432, grid=grid(33554432), stream=stream0)
        buf535 = reinterpret_tensor(buf525, (32, 64, 1024), (65536, 1024, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_656, buf533, out=buf535)
        del permute_656
        buf540 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_32.run(buf535, buf540, 4096, 512, grid=grid(4096, 512), stream=stream0)
        buf542 = reinterpret_tensor(buf535, (4096, 512), (512, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf540, permute_667, out=buf542)
        del permute_667
        buf536 = reinterpret_tensor(buf434, (32, 1024, 64), (65536, 64, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf533, permute_657, out=buf536)
        del permute_657
        buf543 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf536, buf543, 2097152, grid=grid(2097152), stream=stream0)
        buf545 = reinterpret_tensor(buf536, (4096, 512), (512, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf543, permute_672, out=buf545)
        del permute_672
        buf483 = buf474; del buf474  # reuse
        buf510 = buf447; del buf447  # reuse
        buf519 = buf438; del buf438  # reuse
        buf546 = buf411; del buf411  # reuse
        # Source Nodes: [hidden_states_27, hidden_states_31, hidden_states_32, hidden_states_39, hidden_states_40, hidden_states_44, hidden_states_45], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_33.run(buf482, buf0, mm_15, mm_17, mm_21, rsqrt_7, buf503, buf506, buf509, rsqrt_6, buf518, rsqrt_5, buf539, buf542, buf545, rsqrt_4, buf483, buf510, buf519, buf546, 16384, 128, grid=grid(16384), stream=stream0)
        del buf482
        del buf503
        del buf506
        del buf509
        del buf518
        del mm_15
        del mm_17
        del mm_21
        del rsqrt_5
        del rsqrt_6
        del rsqrt_7
        buf484 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_45], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf483, buf484, 512, 32, grid=grid(512), stream=stream0)
        del buf483
        buf487 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (512, 4096), (1, 512), 0), view_91, out=buf487)
        del buf486
        del view_91
        buf502 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (512, 4096), (1, 512), 0), view_73, out=buf502)
        del buf501
        buf505 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (512, 4096), (1, 512), 0), view_73, out=buf505)
        del buf504
        buf508 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (512, 4096), (1, 512), 0), view_73, out=buf508)
        del view_73
        buf511 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_31, hidden_states_39, hidden_states_40], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf510, buf511, 512, 32, grid=grid(512), stream=stream0)
        buf514 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf513, (512, 4096), (1, 512), 0), view_71, out=buf514)
        del view_71
        buf517 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (2048, 4096), (1, 2048), 0), view_69, out=buf517)
        del view_69
        buf520 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf519, buf520, 512, 32, grid=grid(512), stream=stream0)
        buf523 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf522, (512, 4096), (1, 512), 0), view_67, out=buf523)
        del view_67
        buf538 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf537, (512, 4096), (1, 512), 0), view_49, out=buf538)
        buf541 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (512, 4096), (1, 512), 0), view_49, out=buf541)
        buf544 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf543, (512, 4096), (1, 512), 0), view_49, out=buf544)
        del view_49
        buf547 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf546, buf547, 512, 32, grid=grid(512), stream=stream0)
        buf549 = buf522; del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_34.run(buf549, buf539, buf542, buf545, primals_5, buf0, rsqrt_4, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_5
        del rsqrt_4
        buf550 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (512, 4096), (1, 512), 0), view_47, out=buf550)
        del view_47
        buf551 = reinterpret_tensor(buf516, (4096, 2048), (2048, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (4096, 512), (512, 1), 0), permute_676, out=buf551)
        del permute_676
        buf552 = reinterpret_tensor(buf551, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf552, le_11, 8388608, grid=grid(8388608), stream=stream0)
        del le_11
        buf553 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (2048, 4096), (1, 2048), 0), view_45, out=buf553)
        del view_45
        buf554 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (4096, 2048), (2048, 1), 0), permute_680, out=buf554)
        del permute_680
        buf558 = buf549; del buf549  # reuse
        # Source Nodes: [hidden_states_13, hidden_states_18, hidden_states_5], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_31.run(buf558, buf554, primals_4, embedding, mm_3, mm_5, mm_9, rsqrt_3, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_4
        buf560 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (4096, 512), (512, 1), 0), permute_684, out=buf560)
        del permute_684
        buf561 = reinterpret_tensor(buf539, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf560, buf561, 2097152, grid=grid(2097152), stream=stream0)
        buf562 = reinterpret_tensor(buf560, (32, 1024, 64), (65536, 64, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_687, reinterpret_tensor(buf561, (32, 1024, 64), (65536, 64, 1), 0), out=buf562)
        del permute_687
        buf573 = reinterpret_tensor(buf0, (4096, 512), (512, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf562, buf573, 2097152, grid=grid(2097152), stream=stream0)
        buf575 = reinterpret_tensor(buf562, (4096, 512), (512, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf573, permute_695, out=buf575)
        del permute_695
        buf563 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf561, (32, 1024, 64), (65536, 64, 1), 0), permute_688, out=buf563)
        del permute_688
        buf566 = reinterpret_tensor(buf527, (33554432, ), (1, ), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf566, 33554432, grid=grid(33554432), stream=stream0)
        buf569 = reinterpret_tensor(buf120, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf563, alias_118, buf566, buf569, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_118
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf566, buf569, 33554432, grid=grid(33554432), stream=stream0)
        buf571 = reinterpret_tensor(buf561, (32, 64, 1024), (65536, 1024, 1), 0); del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_689, buf569, out=buf571)
        del permute_689
        buf576 = buf543; del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_32.run(buf571, buf576, 4096, 512, grid=grid(4096, 512), stream=stream0)
        buf578 = reinterpret_tensor(buf571, (4096, 512), (512, 1), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf576, permute_700, out=buf578)
        del permute_700
        buf572 = reinterpret_tensor(buf540, (32, 1024, 64), (65536, 64, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf569, permute_690, out=buf572)
        del permute_690
        buf579 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf572, buf579, 2097152, grid=grid(2097152), stream=stream0)
        buf581 = reinterpret_tensor(buf572, (4096, 512), (512, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf579, permute_705, out=buf581)
        del permute_705
        buf585 = buf513; del buf513  # reuse
        # Source Nodes: [hidden_states_13, hidden_states_5], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_16.run(buf575, buf578, buf581, primals_3, embedding, mm_3, mm_5, buf558, rsqrt_2, buf585, 4096, 512, grid=grid(4096), stream=stream0)
        del primals_3
        buf587 = reinterpret_tensor(buf552, (4096, 2048), (2048, 1), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (4096, 512), (512, 1), 0), permute_709, out=buf587)
        del permute_709
        buf588 = reinterpret_tensor(buf587, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_1.run(buf588, le_12, 8388608, grid=grid(8388608), stream=stream0)
        del le_12
        buf590 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (4096, 2048), (2048, 1), 0), permute_713, out=buf590)
        del permute_713
        buf555 = buf546; del buf546  # reuse
        buf582 = buf519; del buf519  # reuse
        buf591 = buf510; del buf510  # reuse
        # Source Nodes: [hidden_states_13, hidden_states_14, hidden_states_18, hidden_states_19, hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_35.run(buf554, embedding, mm_3, mm_5, mm_9, rsqrt_3, buf575, buf578, buf581, rsqrt_2, buf590, rsqrt_1, buf555, buf582, buf591, 16384, 128, grid=grid(16384), stream=stream0)
        del buf554
        del buf575
        del buf578
        del buf581
        del mm_5
        del mm_9
        del rsqrt_2
        del rsqrt_3
        buf556 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13, hidden_states_18, hidden_states_19, hidden_states_5], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf555, buf556, 512, 32, grid=grid(512), stream=stream0)
        del buf555
        buf559 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (512, 4096), (1, 512), 0), view_43, out=buf559)
        del buf558
        del view_43
        buf574 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (512, 4096), (1, 512), 0), view_25, out=buf574)
        buf577 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (512, 4096), (1, 512), 0), view_25, out=buf577)
        buf580 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf579, (512, 4096), (1, 512), 0), view_25, out=buf580)
        del view_25
        buf583 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13, hidden_states_14, hidden_states_5], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf582, buf583, 512, 32, grid=grid(512), stream=stream0)
        del buf582
        buf586 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (512, 4096), (1, 512), 0), view_23, out=buf586)
        del view_23
        buf589 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (2048, 4096), (1, 2048), 0), view_21, out=buf589)
        del view_21
        buf592 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5, hidden_states_6], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf591, buf592, 512, 32, grid=grid(512), stream=stream0)
        buf594 = buf585; del buf585  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_24.run(buf594, buf590, primals_2, embedding, mm_3, rsqrt_1, 4096, 512, grid=grid(4096), stream=stream0)
        del mm_3
        del primals_2
        del rsqrt_1
        buf595 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf594, (512, 4096), (1, 512), 0), view_19, out=buf595)
        del view_19
        buf596 = buf590; del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf594, (4096, 512), (512, 1), 0), permute_717, out=buf596)
        del permute_717
        buf597 = reinterpret_tensor(buf579, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf596, buf597, 2097152, grid=grid(2097152), stream=stream0)
        buf598 = reinterpret_tensor(buf596, (32, 1024, 64), (65536, 64, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_720, reinterpret_tensor(buf597, (32, 1024, 64), (65536, 64, 1), 0), out=buf598)
        del permute_720
        buf599 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf597, (32, 1024, 64), (65536, 64, 1), 0), permute_721, out=buf599)
        del permute_721
        buf602 = reinterpret_tensor(buf563, (33554432, ), (1, ), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_4.run(buf602, 33554432, grid=grid(33554432), stream=stream0)
        buf605 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.as_strided_scatter]
        triton_per_fused__softmax_backward_data_as_strided_scatter_5.run(buf599, alias_122, buf602, buf605, 32768, 1024, grid=grid(32768), stream=stream0)
        del alias_122
        del buf599
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_6.run(buf602, buf605, 33554432, grid=grid(33554432), stream=stream0)
        buf607 = reinterpret_tensor(buf588, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf588  # reuse
        buf609 = reinterpret_tensor(buf607, (1024, 1024, 8), (1024, 1, 1048576), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.sum, aten.threshold_backward]
        triton_poi_fused_add_embedding_dense_backward_sum_threshold_backward_25.run(buf609, buf422, buf458, buf494, buf530, buf566, buf602, 8388608, grid=grid(8388608), stream=stream0)
        del buf422
        del buf458
        del buf494
        del buf530
        del buf566
        del buf602
        buf608 = empty((32, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_threshold_backward_26.run(buf608, 256, grid=grid(256), stream=stream0)
        aten.index_put_(buf608, [add_3], buf609, True)
        del add_3
        del buf609
        buf612 = reinterpret_tensor(buf597, (32, 64, 1024), (65536, 1024, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_723, buf605, out=buf612)
        del permute_723
        buf613 = reinterpret_tensor(buf576, (32, 1024, 64), (65536, 64, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf605, permute_724, out=buf613)
        del buf605
        del permute_724
        buf614 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf598, buf614, 2097152, grid=grid(2097152), stream=stream0)
        buf615 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf614, (512, 4096), (1, 512), 0), view_1, out=buf615)
        buf616 = reinterpret_tensor(buf598, (4096, 512), (512, 1), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf614, permute_729, out=buf616)
        del permute_729
        buf617 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_32.run(buf612, buf617, 4096, 512, grid=grid(4096, 512), stream=stream0)
        buf618 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf617, (512, 4096), (1, 512), 0), view_1, out=buf618)
        buf619 = reinterpret_tensor(buf612, (4096, 512), (512, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf617, permute_734, out=buf619)
        del permute_734
        buf620 = buf617; del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf613, buf620, 2097152, grid=grid(2097152), stream=stream0)
        buf621 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf620, (512, 4096), (1, 512), 0), view_1, out=buf621)
        del view_1
        buf622 = reinterpret_tensor(buf613, (4096, 512), (512, 1), 0); del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf620, permute_739, out=buf622)
        del buf620
        del permute_739
        buf623 = buf591; del buf591  # reuse
        # Source Nodes: [hidden_states_1], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_27.run(buf616, buf619, buf622, embedding, rsqrt, buf623, 16384, 128, grid=grid(16384), stream=stream0)
        buf624 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_19.run(buf623, buf624, 512, 32, grid=grid(512), stream=stream0)
        del buf623
        buf626 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.pow, aten.sum, aten.threshold_backward]
        triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_28.run(buf626, buf616, buf619, buf622, primals_1, embedding, view, rsqrt, 4096, 512, grid=grid(4096), stream=stream0)
        del buf616
        del buf619
        del buf622
        del embedding
        del primals_1
        del rsqrt
        buf627 = empty((32128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_29.run(buf627, 16449536, grid=grid(16449536), stream=stream0)
        aten.index_put_(buf627, [view], buf626, True)
        del buf626
        del view
        buf401 = empty((32128, 512), device='cuda', dtype=torch.float32)
        buf630 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_36.run(buf630, buf399, buf627, 16449536, grid=grid(16449536), stream=stream0)
        return (reinterpret_tensor(buf624, (512, ), (1, ), 0), reinterpret_tensor(buf592, (512, ), (1, ), 0), reinterpret_tensor(buf583, (512, ), (1, ), 0), reinterpret_tensor(buf556, (512, ), (1, ), 0), reinterpret_tensor(buf547, (512, ), (1, ), 0), reinterpret_tensor(buf520, (512, ), (1, ), 0), reinterpret_tensor(buf511, (512, ), (1, ), 0), reinterpret_tensor(buf484, (512, ), (1, ), 0), reinterpret_tensor(buf475, (512, ), (1, ), 0), reinterpret_tensor(buf448, (512, ), (1, ), 0), reinterpret_tensor(buf439, (512, ), (1, ), 0), reinterpret_tensor(buf412, (512, ), (1, ), 0), reinterpret_tensor(buf403, (512, ), (1, ), 0), reinterpret_tensor(buf396, (512, ), (1, ), 0), reinterpret_tensor(buf364, (512, ), (1, ), 0), reinterpret_tensor(buf336, (512, ), (1, ), 0), reinterpret_tensor(buf327, (512, ), (1, ), 0), reinterpret_tensor(buf300, (512, ), (1, ), 0), reinterpret_tensor(buf273, (512, ), (1, ), 0), reinterpret_tensor(buf264, (512, ), (1, ), 0), reinterpret_tensor(buf237, (512, ), (1, ), 0), reinterpret_tensor(buf209, (512, ), (1, ), 0), reinterpret_tensor(buf200, (512, ), (1, ), 0), reinterpret_tensor(buf173, (512, ), (1, ), 0), reinterpret_tensor(buf146, (512, ), (1, ), 0), reinterpret_tensor(buf137, (512, ), (1, ), 0), reinterpret_tensor(buf110, (512, ), (1, ), 0), reinterpret_tensor(buf83, (512, ), (1, ), 0), reinterpret_tensor(buf74, (512, ), (1, ), 0), reinterpret_tensor(buf47, (512, ), (1, ), 0), reinterpret_tensor(buf19, (512, ), (1, ), 0), reinterpret_tensor(buf10, (512, ), (1, ), 0), buf630, reinterpret_tensor(buf621, (512, 512), (512, 1), 0), reinterpret_tensor(buf618, (512, 512), (512, 1), 0), reinterpret_tensor(buf615, (512, 512), (512, 1), 0), buf608, reinterpret_tensor(buf595, (512, 512), (512, 1), 0), reinterpret_tensor(buf589, (2048, 512), (512, 1), 0), reinterpret_tensor(buf586, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf580, (512, 512), (512, 1), 0), reinterpret_tensor(buf577, (512, 512), (512, 1), 0), reinterpret_tensor(buf574, (512, 512), (512, 1), 0), reinterpret_tensor(buf559, (512, 512), (512, 1), 0), reinterpret_tensor(buf553, (2048, 512), (512, 1), 0), reinterpret_tensor(buf550, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf544, (512, 512), (512, 1), 0), reinterpret_tensor(buf541, (512, 512), (512, 1), 0), reinterpret_tensor(buf538, (512, 512), (512, 1), 0), reinterpret_tensor(buf523, (512, 512), (512, 1), 0), reinterpret_tensor(buf517, (2048, 512), (512, 1), 0), reinterpret_tensor(buf514, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf508, (512, 512), (512, 1), 0), reinterpret_tensor(buf505, (512, 512), (512, 1), 0), reinterpret_tensor(buf502, (512, 512), (512, 1), 0), reinterpret_tensor(buf487, (512, 512), (512, 1), 0), reinterpret_tensor(buf481, (2048, 512), (512, 1), 0), reinterpret_tensor(buf478, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf472, (512, 512), (512, 1), 0), reinterpret_tensor(buf469, (512, 512), (512, 1), 0), reinterpret_tensor(buf466, (512, 512), (512, 1), 0), reinterpret_tensor(buf451, (512, 512), (512, 1), 0), reinterpret_tensor(buf445, (2048, 512), (512, 1), 0), reinterpret_tensor(buf442, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf436, (512, 512), (512, 1), 0), reinterpret_tensor(buf433, (512, 512), (512, 1), 0), reinterpret_tensor(buf430, (512, 512), (512, 1), 0), reinterpret_tensor(buf415, (512, 512), (512, 1), 0), reinterpret_tensor(buf409, (2048, 512), (512, 1), 0), reinterpret_tensor(buf406, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf393, (512, 512), (512, 1), 0), reinterpret_tensor(buf390, (512, 512), (512, 1), 0), reinterpret_tensor(buf387, (512, 512), (512, 1), 0), buf380, reinterpret_tensor(buf367, (512, 512), (512, 1), 0), reinterpret_tensor(buf361, (512, 512), (512, 1), 0), reinterpret_tensor(buf357, (512, 512), (512, 1), 0), reinterpret_tensor(buf354, (512, 512), (512, 1), 0), reinterpret_tensor(buf339, (512, 512), (512, 1), 0), reinterpret_tensor(buf333, (2048, 512), (512, 1), 0), reinterpret_tensor(buf330, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf324, (512, 512), (512, 1), 0), reinterpret_tensor(buf321, (512, 512), (512, 1), 0), reinterpret_tensor(buf318, (512, 512), (512, 1), 0), reinterpret_tensor(buf303, (512, 512), (512, 1), 0), reinterpret_tensor(buf297, (512, 512), (512, 1), 0), reinterpret_tensor(buf294, (512, 512), (512, 1), 0), reinterpret_tensor(buf291, (512, 512), (512, 1), 0), reinterpret_tensor(buf276, (512, 512), (512, 1), 0), reinterpret_tensor(buf270, (2048, 512), (512, 1), 0), reinterpret_tensor(buf267, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf261, (512, 512), (512, 1), 0), reinterpret_tensor(buf258, (512, 512), (512, 1), 0), reinterpret_tensor(buf255, (512, 512), (512, 1), 0), reinterpret_tensor(buf240, (512, 512), (512, 1), 0), reinterpret_tensor(buf234, (512, 512), (512, 1), 0), reinterpret_tensor(buf230, (512, 512), (512, 1), 0), reinterpret_tensor(buf227, (512, 512), (512, 1), 0), reinterpret_tensor(buf212, (512, 512), (512, 1), 0), reinterpret_tensor(buf206, (2048, 512), (512, 1), 0), reinterpret_tensor(buf203, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf197, (512, 512), (512, 1), 0), reinterpret_tensor(buf194, (512, 512), (512, 1), 0), reinterpret_tensor(buf191, (512, 512), (512, 1), 0), reinterpret_tensor(buf176, (512, 512), (512, 1), 0), reinterpret_tensor(buf170, (512, 512), (512, 1), 0), reinterpret_tensor(buf167, (512, 512), (512, 1), 0), reinterpret_tensor(buf164, (512, 512), (512, 1), 0), reinterpret_tensor(buf149, (512, 512), (512, 1), 0), reinterpret_tensor(buf143, (2048, 512), (512, 1), 0), reinterpret_tensor(buf140, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf134, (512, 512), (512, 1), 0), reinterpret_tensor(buf131, (512, 512), (512, 1), 0), reinterpret_tensor(buf128, (512, 512), (512, 1), 0), reinterpret_tensor(buf113, (512, 512), (512, 1), 0), reinterpret_tensor(buf107, (512, 512), (512, 1), 0), reinterpret_tensor(buf104, (512, 512), (512, 1), 0), reinterpret_tensor(buf101, (512, 512), (512, 1), 0), reinterpret_tensor(buf86, (512, 512), (512, 1), 0), reinterpret_tensor(buf80, (2048, 512), (512, 1), 0), reinterpret_tensor(buf77, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf71, (512, 512), (512, 1), 0), reinterpret_tensor(buf68, (512, 512), (512, 1), 0), reinterpret_tensor(buf65, (512, 512), (512, 1), 0), reinterpret_tensor(buf50, (512, 512), (512, 1), 0), reinterpret_tensor(buf44, (512, 512), (512, 1), 0), reinterpret_tensor(buf41, (512, 512), (512, 1), 0), reinterpret_tensor(buf38, (512, 512), (512, 1), 0), reinterpret_tensor(buf22, (512, 512), (512, 1), 0), reinterpret_tensor(buf16, (2048, 512), (512, 1), 0), reinterpret_tensor(buf13, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf7, (32128, 512), (512, 1), 0), None, None, )


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
    view = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((4, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_3 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    view_19 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_9 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_11 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_4 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_15 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_17 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_21 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_23 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_8 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_27 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_29 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_33 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_35 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_12 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    embedding_2 = rand_strided((4, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_37 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    view_164 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_39 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_14 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_169 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_43 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_186 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_188 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_45 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_208 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_49 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_17 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_210 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_228 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_53 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_232 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_55 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_234 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_252 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_59 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_254 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_272 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_63 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_21 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_276 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_65 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_22 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_278 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_296 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_69 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_23 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_298 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_316 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_73 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_24 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_318 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_320 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_75 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_322 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_340 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_79 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_26 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_342 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_360 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_83 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_27 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_362 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_364 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_85 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_28 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_366 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_384 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_89 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_29 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_386 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_404 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_93 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_30 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_406 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_408 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mm_95 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_31 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_410 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_199 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_65 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_67 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_239 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_257 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_71 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_272 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_73 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_292 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_297 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_302 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_315 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_77 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_330 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_347 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_79 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_373 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_380 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_83 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_85 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_407 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_423 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_431 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_89 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_441 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_451 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_456 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_91 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_466 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_489 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_493 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_95 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_499 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_504 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_97 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_524 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_540 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_544 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_548 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_552 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_556 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_102 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_557 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_563 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_568 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_573 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_577 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_581 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_585 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_588 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_589 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_106 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_596 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_601 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_606 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_610 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_614 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_618 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_621 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_622 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_110 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_623 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_629 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_634 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_639 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_643 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_10 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_647 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_651 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_654 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_655 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_114 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_656 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_657 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_662 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_667 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_672 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_676 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_680 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_684 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_688 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_118 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_689 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_690 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_695 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_700 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_705 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_709 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cuda:0', dtype=torch.bool)
    permute_713 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_720 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_122 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_723 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_724 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_734 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_739 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1024, 32128), (32899072, 32128, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((4, 1024, 512), (524288, 512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, view, embedding, rsqrt, view_1, add_3, view_19, mm_3, rsqrt_1, view_21, view_23, mm_5, rsqrt_2, view_25, view_43, mm_9, rsqrt_3, view_45, view_47, mm_11, rsqrt_4, view_49, view_67, mm_15, rsqrt_5, view_69, view_71, mm_17, rsqrt_6, view_73, view_91, mm_21, rsqrt_7, view_93, view_95, mm_23, rsqrt_8, view_97, view_115, mm_27, rsqrt_9, view_117, view_119, mm_29, rsqrt_10, view_121, view_139, mm_33, rsqrt_11, view_141, view_143, mm_35, rsqrt_12, view_145, embedding_2, rsqrt_13, view_146, add_37, view_164, mm_39, rsqrt_14, view_166, view_169, view_184, mm_43, rsqrt_15, view_186, view_188, mm_45, rsqrt_16, view_190, view_208, mm_49, rsqrt_17, view_210, view_228, mm_53, rsqrt_18, view_230, view_232, mm_55, rsqrt_19, view_234, view_252, mm_59, rsqrt_20, view_254, view_272, mm_63, rsqrt_21, view_274, view_276, mm_65, rsqrt_22, view_278, view_296, mm_69, rsqrt_23, view_298, view_316, mm_73, rsqrt_24, view_318, view_320, mm_75, rsqrt_25, view_322, view_340, mm_79, rsqrt_26, view_342, view_360, mm_83, rsqrt_27, view_362, view_364, mm_85, rsqrt_28, view_366, view_384, mm_89, rsqrt_29, view_386, view_404, mm_93, rsqrt_30, view_406, view_408, mm_95, rsqrt_31, view_410, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_65, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_67, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_71, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_73, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_77, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_79, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_83, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_85, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_89, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_91, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_95, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_97, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_102, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_106, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_110, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_114, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_118, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_122, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_T5', benchmark_compiled_module)
