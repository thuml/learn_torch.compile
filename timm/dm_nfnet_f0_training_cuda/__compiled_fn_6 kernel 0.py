
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


# kernel path: /tmp/torchinductor_youkaichao/ca/ccaivxb7qitqbda33lu6gqtyz6oejydkkqkjcbhwgdjuubua763z.py
# Source Nodes: [gelu_51], Original ATen: [aten.div, aten.gelu, aten.gelu_backward, aten.mul]
# gelu_51 => add_120, erf_51, mul_436
triton_poi_fused_div_gelu_gelu_backward_mul_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_gelu_gelu_backward_mul_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 36)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp1 = 36.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1.7015043497085571
    tmp4 = tmp2 * tmp3
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp4 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvix5twdks2ch3k5td2x54dvegfqvl6aqo6xnbpmcc5e7bg3vzl.py
# Source Nodes: [sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_11 => sigmoid_11
triton_per_fused_mul_sigmoid_sigmoid_backward_sum_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_sum_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.load(in_ptr2 + (r1 + (36*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = 1.0
    tmp17 = tmp16 - tmp15
    tmp18 = tmp15 * tmp17
    tmp19 = tmp13 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uy/cuybn5m6mfskyd3qyjqddc23eoemy2pylofixkrpyk3ip4ad4yg2.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4cvnzxpxfmb3mwfaqhoutlfdayeve5cb4mm5j6bia7os5awh7xs.py
# Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_11 => sigmoid_11
triton_poi_fused_add_div_mul_sigmoid_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp7 * tmp9
    tmp12 = 36.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vlso44tkg34jsdruca23ro2goplsp6szbyfb5e6lrxuda7ehin.py
# Source Nodes: [gelu_50], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
# gelu_50 => add_116, erf_50, mul_422
triton_poi_fused_gelu_gelu_backward_mul_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_mul_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.7015043497085571
    tmp2 = tmp0 * tmp1
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp2 * tmp18
    tl.store(in_out_ptr0 + (x0), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqnhxpvon5finiouq56bescetwifxi7wo6kykfatkf6szkp2zjf.py
# Source Nodes: [gelu_47, mul_85, mul_87, mul_93, mul_95, mul__52, mul__57, out_77, out_85, shortcut_14, shortcut_15, sigmoid_10, sigmoid_9], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_47 => add_110, erf_47, mul_400
# mul_85 => mul_362
# mul_87 => mul_365
# mul_93 => mul_395
# mul_95 => mul_398
# mul__52 => mul_364
# mul__57 => mul_397
# out_77 => mul_363
# out_85 => mul_396
# shortcut_14 => add_100
# shortcut_15 => add_109
# sigmoid_10 => sigmoid_10
# sigmoid_9 => sigmoid_9
triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (36*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr6 + (r1 + (36*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr7 + (r1 + (36*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr8 + (r1 + (36*x0)), rmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp25 = 0.9622504486493761
    tmp26 = tmp24 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.7071067811865476
    tmp30 = tmp22 * tmp29
    tmp31 = tl.math.erf(tmp30)
    tmp32 = 1.0
    tmp33 = tmp31 + tmp32
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = tmp22 * tmp22
    tmp37 = -0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tl.exp(tmp38)
    tmp40 = 0.3989422804014327
    tmp41 = tmp39 * tmp40
    tmp42 = tmp22 * tmp41
    tmp43 = tmp35 + tmp42
    tmp44 = tmp28 * tmp43
    tmp45 = tmp23 + tmp44
    tmp46 = tmp45 * tmp9
    tmp47 = tmp46 * tmp7
    tmp48 = tmp47 * tmp4
    tmp49 = tmp48 * tmp0
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
    tmp52 = tl.where(rmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp54 = tmp32 - tmp2
    tmp55 = tmp2 * tmp54
    tmp56 = tmp53 * tmp55
    tl.store(out_ptr0 + (r1 + (36*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp56, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qy/cqytba5f3y4nix7ufu2mmvquo5e6bpsd5hlkdj7o53vv6r7loml2.py
# Source Nodes: [gelu_47, sigmoid_10], Original ATen: [aten.add, aten.div, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid]
# gelu_47 => add_110, erf_47, mul_400
# sigmoid_10 => sigmoid_10
triton_poi_fused_add_div_gelu_gelu_backward_mul_sigmoid_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_gelu_gelu_backward_mul_sigmoid_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp26 = tl.load(in_ptr3 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp31 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp4 = 1.7015043497085571
    tmp5 = tmp3 * tmp4
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
    tmp23 = tmp0 + tmp22
    tmp24 = 0.2
    tmp25 = tmp23 * tmp24
    tmp28 = tmp25 * tmp27
    tmp29 = 2.0
    tmp30 = tmp28 * tmp29
    tmp32 = tl.sigmoid(tmp31)
    tmp33 = tmp30 * tmp32
    tmp35 = 36.0
    tmp36 = tmp34 / tmp35
    tmp37 = tmp33 + tmp36
    tl.store(out_ptr0 + (x2), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxcxolzktwb32rki4thazrgjdbukrzzzonawcrcyvro56tsponp.py
# Source Nodes: [gelu_43, gelu_47, mul_85, mul_87, mul__52, out_77, shortcut_14, sigmoid_9], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_43 => add_101, erf_43, mul_367
# gelu_47 => add_110, erf_47, mul_400
# mul_85 => mul_362
# mul_87 => mul_365
# mul__52 => mul_364
# out_77 => mul_363
# shortcut_14 => add_100
# sigmoid_9 => sigmoid_9
triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (36*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr4 + (r1 + (36*x0)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr5 + (r1 + (36*x0)), rmask, other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1 + (36*x0)), rmask, other=0.0)
    tmp47 = tl.load(in_out_ptr0 + (r1 + (36*x0)), rmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 0.7071067811865476
    tmp14 = tmp12 * tmp13
    tmp15 = tl.math.erf(tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.5
    tmp19 = tmp17 * tmp18
    tmp20 = tmp12 * tmp12
    tmp21 = -0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tl.exp(tmp22)
    tmp24 = 0.3989422804014327
    tmp25 = tmp23 * tmp24
    tmp26 = tmp12 * tmp25
    tmp27 = tmp19 + tmp26
    tmp30 = 0.9622504486493761
    tmp31 = tmp29 * tmp30
    tmp32 = 1.7015043497085571
    tmp33 = tmp31 * tmp32
    tmp35 = tmp34 * tmp13
    tmp36 = tl.math.erf(tmp35)
    tmp37 = tmp36 + tmp16
    tmp38 = tmp37 * tmp18
    tmp39 = tmp34 * tmp34
    tmp40 = tmp39 * tmp21
    tmp41 = tl.exp(tmp40)
    tmp42 = tmp41 * tmp24
    tmp43 = tmp34 * tmp42
    tmp44 = tmp38 + tmp43
    tmp45 = tmp33 * tmp44
    tmp46 = tmp28 + tmp45
    tmp48 = 0.9805806756909201
    tmp49 = tmp47 * tmp48
    tmp50 = tmp49 * tmp32
    tmp51 = tmp50 * tmp27
    tmp52 = tmp46 + tmp51
    tmp53 = tmp52 * tmp9
    tmp54 = tmp53 * tmp7
    tmp55 = tmp54 * tmp4
    tmp56 = tmp55 * tmp0
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.where(rmask, tmp57, 0)
    tmp60 = tl.sum(tmp59, 1)[:, None]
    tmp61 = tmp16 - tmp2
    tmp62 = tmp2 * tmp61
    tmp63 = tmp60 * tmp62
    tl.store(in_out_ptr0 + (r1 + (36*x0)), tmp52, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp63, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cww7y7rwyxbx3lf6qfzeh2z6oswnnlx4pqaknb7ypj4ponzxlrsq.py
# Source Nodes: [gelu_40], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
# gelu_40 => add_94, erf_40, mul_342
triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 12) % 12
    x0 = xindex % 12
    x2 = (xindex // 144)
    x3 = xindex
    tmp11 = tl.load(in_ptr1 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 13, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (13*x1) + (169*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1.7015043497085571
    tmp10 = tmp8 * tmp9
    tmp12 = 0.7071067811865476
    tmp13 = tmp11 * tmp12
    tmp14 = tl.math.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tmp11 * tmp11
    tmp20 = -0.5
    tmp21 = tmp19 * tmp20
    tmp22 = tl.exp(tmp21)
    tmp23 = 0.3989422804014327
    tmp24 = tmp22 * tmp23
    tmp25 = tmp11 * tmp24
    tmp26 = tmp18 + tmp25
    tmp27 = tmp10 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv55bs7ewdsuvxur66okjrfv4f3757bissdtzjbf2jwkgvcw4zyz.py
# Source Nodes: [gelu_27, gelu_35, gelu_39, mul_36, mul_38, mul_44, mul_46, mul_52, mul_54, mul_60, mul_62, mul_68, mul_70, mul_76, mul_78, mul__22, mul__27, mul__32, mul__37, mul__42, mul__47, out_29, out_37, out_45, out_53, out_61, out_69, shortcut_10, shortcut_11, shortcut_12, shortcut_7, shortcut_8, shortcut_9, sigmoid_3, sigmoid_4, sigmoid_5, sigmoid_6, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_27 => add_64, erf_27, mul_232
# gelu_35 => add_82, erf_35, mul_298
# gelu_39 => add_91, erf_39, mul_331
# mul_36 => mul_161
# mul_38 => mul_164
# mul_44 => mul_194
# mul_46 => mul_197
# mul_52 => mul_227
# mul_54 => mul_230
# mul_60 => mul_260
# mul_62 => mul_263
# mul_68 => mul_293
# mul_70 => mul_296
# mul_76 => mul_326
# mul_78 => mul_329
# mul__22 => mul_163
# mul__27 => mul_196
# mul__32 => mul_229
# mul__37 => mul_262
# mul__42 => mul_295
# mul__47 => mul_328
# out_29 => mul_162
# out_37 => mul_195
# out_45 => mul_228
# out_53 => mul_261
# out_61 => mul_294
# out_69 => mul_327
# shortcut_10 => add_72
# shortcut_11 => add_81
# shortcut_12 => add_90
# shortcut_7 => add_45
# shortcut_8 => add_54
# shortcut_9 => add_63
# sigmoid_3 => sigmoid_3
# sigmoid_4 => sigmoid_4
# sigmoid_5 => sigmoid_5
# sigmoid_6 => sigmoid_6
# sigmoid_7 => sigmoid_7
# sigmoid_8 => sigmoid_8
triton_red_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: 'i32', 27: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(26, 27))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, out_ptr1, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr9 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp33 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr12 + (0))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp44 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr15 + (0))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp53 = tl.load(in_ptr17 + (x0), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr18 + (0))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    _tmp121 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        r2 = rindex % 12
        r3 = (rindex // 12)
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr6 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr7 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr10 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr13 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp52 = tl.load(in_ptr16 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp64 = tl.load(in_ptr19 + ((6*(tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(6, 1 + (r3 // 2)))))) + (6*(tl.where((tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(6, 1 + (r3 // 2))))) >= 0, 0, 6))) + (36*x0) + (tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(6, 1 + (r2 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(6, 1 + (r2 // 2))))) >= 0, 0, 6))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp8 = tmp5 * tmp7
        tmp9 = 0.2
        tmp10 = tmp8 * tmp9
        tmp13 = tl.sigmoid(tmp12)
        tmp14 = tmp11 * tmp13
        tmp15 = tmp14 * tmp4
        tmp18 = tmp15 * tmp17
        tmp19 = tmp18 * tmp9
        tmp21 = tmp19 + tmp20
        tmp22 = tmp10 + tmp21
        tmp25 = tl.sigmoid(tmp24)
        tmp26 = tmp23 * tmp25
        tmp27 = tmp26 * tmp4
        tmp30 = tmp27 * tmp29
        tmp31 = tmp30 * tmp9
        tmp34 = tl.sigmoid(tmp33)
        tmp35 = tmp32 * tmp34
        tmp36 = tmp35 * tmp4
        tmp39 = tmp36 * tmp38
        tmp40 = tmp39 * tmp9
        tmp41 = tmp40 + tmp22
        tmp42 = tmp31 + tmp41
        tmp45 = tl.sigmoid(tmp44)
        tmp46 = tmp43 * tmp45
        tmp47 = tmp46 * tmp4
        tmp50 = tmp47 * tmp49
        tmp51 = tmp50 * tmp9
        tmp54 = tl.sigmoid(tmp53)
        tmp55 = tmp52 * tmp54
        tmp56 = tmp55 * tmp4
        tmp59 = tmp56 * tmp58
        tmp60 = tmp59 * tmp9
        tmp61 = tmp60 + tmp42
        tmp62 = tmp51 + tmp61
        tmp65 = tmp64 / 4
        tmp66 = tl.math.max(0, (r3 // 2))
        tmp67 = tl.math.min(6, 1 + (r3 // 2))
        tmp68 = tmp66 < tmp67
        tmp69 = tl.math.max(0, (r2 // 2))
        tmp70 = tl.math.min(6, 1 + (r2 // 2))
        tmp71 = tmp69 < tmp70
        tmp72 = tmp68 & tmp71
        tmp73 = 0.0
        tmp74 = tl.where(tmp72, tmp65, tmp73)
        tmp75 = tmp63 + tmp74
        tmp76 = 0.8980265101338745
        tmp77 = tmp75 * tmp76
        tmp78 = 1.7015043497085571
        tmp79 = tmp77 * tmp78
        tmp80 = 0.7071067811865476
        tmp81 = tmp62 * tmp80
        tmp82 = tl.math.erf(tmp81)
        tmp83 = 1.0
        tmp84 = tmp82 + tmp83
        tmp85 = 0.5
        tmp86 = tmp84 * tmp85
        tmp87 = tmp62 * tmp62
        tmp88 = -0.5
        tmp89 = tmp87 * tmp88
        tmp90 = tl.exp(tmp89)
        tmp91 = 0.3989422804014327
        tmp92 = tmp90 * tmp91
        tmp93 = tmp62 * tmp92
        tmp94 = tmp86 + tmp93
        tmp95 = tmp79 * tmp94
        tmp96 = tmp61 * tmp80
        tmp97 = tl.math.erf(tmp96)
        tmp98 = tmp97 + tmp83
        tmp99 = tmp98 * tmp85
        tmp100 = tmp61 * tmp61
        tmp101 = tmp100 * tmp88
        tmp102 = tl.exp(tmp101)
        tmp103 = tmp102 * tmp91
        tmp104 = tmp61 * tmp103
        tmp105 = tmp99 + tmp104
        tmp106 = tmp41 * tmp80
        tmp107 = tl.math.erf(tmp106)
        tmp108 = tmp107 + tmp83
        tmp109 = tmp108 * tmp85
        tmp110 = tmp41 * tmp41
        tmp111 = tmp110 * tmp88
        tmp112 = tl.exp(tmp111)
        tmp113 = tmp112 * tmp91
        tmp114 = tmp41 * tmp113
        tmp115 = tmp109 + tmp114
        tmp116 = tmp95 * tmp9
        tmp117 = tmp116 * tmp49
        tmp118 = tmp117 * tmp4
        tmp119 = tmp118 * tmp43
        tmp120 = tl.broadcast_to(tmp119, [XBLOCK, RBLOCK])
        tmp122 = _tmp121 + tmp120
        _tmp121 = tl.where(rmask, tmp122, _tmp121)
        tl.store(out_ptr0 + (r1 + (144*x0)), tmp22, rmask)
        tl.store(out_ptr1 + (r1 + (144*x0)), tmp42, rmask)
        tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp95, rmask)
        tl.store(out_ptr3 + (r1 + (144*x0)), tmp105, rmask)
        tl.store(out_ptr4 + (r1 + (144*x0)), tmp115, rmask)
    tmp121 = tl.sum(_tmp121, 1)[:, None]
    tmp123 = tl.sigmoid(tmp44)
    tmp124 = 1.0
    tmp125 = tmp124 - tmp123
    tmp126 = tmp123 * tmp125
    tmp127 = tmp121 * tmp126
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp127, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clkqcz6obdactkl4barqhp2xhh5tbcfqmht7awomvlspxean7bfd.py
# Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_8 => sigmoid_8
triton_poi_fused_add_div_mul_sigmoid_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp7 * tmp9
    tmp12 = 144.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx24ah4wherhx4axylfoz5wearfrx4fkkktlz7qi7ayktfvpct6q.py
# Source Nodes: [gelu_38], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
# gelu_38 => add_88, erf_38, mul_320
triton_poi_fused_gelu_gelu_backward_mul_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_mul_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.7015043497085571
    tmp2 = tmp0 * tmp1
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp2 * tmp18
    tl.store(in_out_ptr0 + (x0), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtvavoc4dvw775r2snlihmj5a5iyuqka62hqfoaxypmahemrg3y.py
# Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_7 => sigmoid_7
triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.9128709291752768
        tmp3 = tmp1 * tmp2
        tmp4 = 1.7015043497085571
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 0.2
        tmp10 = tmp8 * tmp9
        tmp13 = tmp10 * tmp12
        tmp14 = 2.0
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = 1.0
    tmp24 = tmp23 - tmp22
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2w4pucgxq2xldxg57ueyexxbb7tachx7mdj5viyhtbe7d4s7at5.py
# Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.div, aten.gelu_backward, aten.mul, aten.sigmoid]
# sigmoid_7 => sigmoid_7
triton_poi_fused_add_div_gelu_backward_mul_sigmoid_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_gelu_backward_mul_sigmoid_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9128709291752768
    tmp3 = tmp1 * tmp2
    tmp4 = 1.7015043497085571
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tmp10 * tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp15 * tmp17
    tmp20 = 144.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp18 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprdmz2c57cyghv6yu7cqqcp5anmzc74d5h4mrvudlnx6fuyhffk.py
# Source Nodes: [gelu_31, sigmoid_6], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_31 => add_73, erf_31, mul_265
# sigmoid_6 => sigmoid_6
triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp33 = tl.load(in_ptr4 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr5 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.9128709291752768
        tmp3 = tmp1 * tmp2
        tmp4 = 1.7015043497085571
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 + tmp7
        tmp10 = 0.9284766908852592
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11 * tmp4
        tmp14 = 0.7071067811865476
        tmp15 = tmp13 * tmp14
        tmp16 = tl.math.erf(tmp15)
        tmp17 = 1.0
        tmp18 = tmp16 + tmp17
        tmp19 = 0.5
        tmp20 = tmp18 * tmp19
        tmp21 = tmp13 * tmp13
        tmp22 = -0.5
        tmp23 = tmp21 * tmp22
        tmp24 = tl.exp(tmp23)
        tmp25 = 0.3989422804014327
        tmp26 = tmp24 * tmp25
        tmp27 = tmp13 * tmp26
        tmp28 = tmp20 + tmp27
        tmp29 = tmp12 * tmp28
        tmp30 = tmp8 + tmp29
        tmp31 = 0.2
        tmp32 = tmp30 * tmp31
        tmp35 = tmp32 * tmp34
        tmp36 = 2.0
        tmp37 = tmp35 * tmp36
        tmp39 = tmp37 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask, tmp42, _tmp41)
        tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp30, rmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp43 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp44 = tl.sigmoid(tmp43)
    tmp45 = 1.0
    tmp46 = tmp45 - tmp44
    tmp47 = tmp44 * tmp46
    tmp48 = tmp41 * tmp47
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp48, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhfn3r4t5jgjkweusynkh2zdzdld4otz5dufsapwpdbg7pndwmi.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_5 => sigmoid_5
triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.9449111825230679
        tmp3 = tmp1 * tmp2
        tmp4 = 1.7015043497085571
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 0.2
        tmp10 = tmp8 * tmp9
        tmp13 = tmp10 * tmp12
        tmp14 = 2.0
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = 1.0
    tmp24 = tmp23 - tmp22
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qb4jwh5bsp5gwc5kvj7gfgestpvk24vh6zb52la7vzp2hsw3op.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.gelu_backward, aten.mul, aten.sigmoid]
# sigmoid_5 => sigmoid_5
triton_poi_fused_add_div_gelu_backward_mul_sigmoid_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_gelu_backward_mul_sigmoid_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9449111825230679
    tmp3 = tmp1 * tmp2
    tmp4 = 1.7015043497085571
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tmp10 * tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp15 * tmp17
    tmp20 = 144.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp18 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zcycjf5vrnoxbdeptvedsp3iqk6m227ks4tjzpkhk54usnqcbw.py
# Source Nodes: [gelu_23, mul_44, out_37, sigmoid_4], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
# gelu_23 => add_55, erf_23, mul_199
# mul_44 => mul_194
# out_37 => mul_195
# sigmoid_4 => sigmoid_4
triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_out_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr5 + (((r1 + (8192*x0)) // 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.9449111825230679
        tmp3 = tmp1 * tmp2
        tmp4 = 1.7015043497085571
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 + tmp7
        tmp10 = 0.9622504486493761
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11 * tmp4
        tmp14 = 0.7071067811865476
        tmp15 = tmp13 * tmp14
        tmp16 = tl.math.erf(tmp15)
        tmp17 = 1.0
        tmp18 = tmp16 + tmp17
        tmp19 = 0.5
        tmp20 = tmp18 * tmp19
        tmp21 = tmp13 * tmp13
        tmp22 = -0.5
        tmp23 = tmp21 * tmp22
        tmp24 = tl.exp(tmp23)
        tmp25 = 0.3989422804014327
        tmp26 = tmp24 * tmp25
        tmp27 = tmp13 * tmp26
        tmp28 = tmp20 + tmp27
        tmp29 = tmp12 * tmp28
        tmp30 = tmp8 + tmp29
        tmp31 = 0.2
        tmp32 = tmp30 * tmp31
        tmp35 = tl.sigmoid(tmp34)
        tmp36 = tmp33 * tmp35
        tmp37 = 2.0
        tmp38 = tmp36 * tmp37
        tmp39 = tmp32 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
        tl.store(in_out_ptr0 + (r1 + (8192*x0)), tmp30, rmask & xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c656cyilunjwrerxmtv6noklv2g25hughq6vk7u4u32dowryszqm.py
# Source Nodes: [sigmoid_4], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_4 => sigmoid_4
triton_red_fused_mul_sigmoid_sigmoid_backward_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sigmoid_sigmoid_backward_sum_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tmp2 * tmp4
        tmp6 = 2.0
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6wzpc5hg36wtk4uiuq6gkcwvyhkvlv3kcre2ir3sttzuvnqykd.py
# Source Nodes: [gelu_19, mul_36, mul_38, mul__22, out_29, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_19 => add_46, erf_19, mul_166
# mul_36 => mul_161
# mul_38 => mul_164
# mul__22 => mul_163
# out_29 => mul_162
# shortcut_7 => add_45
# sigmoid_3 => sigmoid_3
triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr4 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp8 = tmp5 * tmp7
        tmp9 = 0.2
        tmp10 = tmp8 * tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = 0.7071067811865476
        tmp14 = tmp12 * tmp13
        tmp15 = tl.math.erf(tmp14)
        tmp16 = 1.0
        tmp17 = tmp15 + tmp16
        tmp18 = 0.5
        tmp19 = tmp17 * tmp18
        tmp20 = tmp12 * tmp12
        tmp21 = -0.5
        tmp22 = tmp20 * tmp21
        tmp23 = tl.exp(tmp22)
        tmp24 = 0.3989422804014327
        tmp25 = tmp23 * tmp24
        tmp26 = tmp12 * tmp25
        tmp27 = tmp19 + tmp26
        tmp30 = 0.9805806756909201
        tmp31 = tmp29 * tmp30
        tmp32 = 1.7015043497085571
        tmp33 = tmp31 * tmp32
        tmp34 = tmp33 * tmp27
        tmp35 = tmp28 + tmp34
        tmp36 = tmp35 * tmp9
        tmp37 = tmp36 * tmp7
        tmp38 = tmp37 * tmp4
        tmp39 = tmp38 * tmp0
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask, tmp42, _tmp41)
        tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp35, rmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp43 = tl.sigmoid(tmp1)
    tmp44 = 1.0
    tmp45 = tmp44 - tmp43
    tmp46 = tmp43 * tmp45
    tmp47 = tmp41 * tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csgquxcv6lzhknouwvuzgsuyrhjutmjtdk3cznpy5k3niuje7d6q.py
# Source Nodes: [gelu_16], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
# gelu_16 => add_39, erf_16, mul_141
triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 24) % 24
    x0 = xindex % 24
    x2 = (xindex // 576)
    x3 = xindex
    tmp11 = tl.load(in_ptr1 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (25*x1) + (625*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1.7015043497085571
    tmp10 = tmp8 * tmp9
    tmp12 = 0.7071067811865476
    tmp13 = tmp11 * tmp12
    tmp14 = tl.math.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tmp11 * tmp11
    tmp20 = -0.5
    tmp21 = tmp19 * tmp20
    tmp22 = tl.exp(tmp21)
    tmp23 = 0.3989422804014327
    tmp24 = tmp22 * tmp23
    tmp25 = tmp11 * tmp24
    tmp26 = tmp18 + tmp25
    tmp27 = tmp10 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4xpsmuicnihfesxnkhjuh3cxz5q7nufyagb6n2mmqlfjknydee.py
# Source Nodes: [gelu_15, mul_19, mul_21, mul_27, mul_29, mul__12, mul__17, out_13, out_21, shortcut_4, shortcut_5, sigmoid_1, sigmoid_2], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_15 => add_36, erf_15, mul_130
# mul_19 => mul_92
# mul_21 => mul_95
# mul_27 => mul_125
# mul_29 => mul_128
# mul__12 => mul_94
# mul__17 => mul_127
# out_13 => mul_93
# out_21 => mul_126
# shortcut_4 => add_26
# shortcut_5 => add_35
# sigmoid_1 => sigmoid_1
# sigmoid_2 => sigmoid_2
triton_per_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    r2 = rindex % 24
    r3 = (rindex // 24)
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (576*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp20 = tl.load(in_ptr6 + (r1 + (576*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr7 + (r1 + (576*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr8 + ((12*(tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(12, 1 + (r3 // 2)))))) + (12*(tl.where((tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(12, 1 + (r3 // 2))))) >= 0, 0, 12))) + (144*x0) + (tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(12, 1 + (r2 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(12, 1 + (r2 // 2))))) >= 0, 0, 12))), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp25 = tmp24 / 4
    tmp26 = tl.math.max(0, (r3 // 2))
    tmp27 = tl.math.min(12, 1 + (r3 // 2))
    tmp28 = tmp26 < tmp27
    tmp29 = tl.math.max(0, (r2 // 2))
    tmp30 = tl.math.min(12, 1 + (r2 // 2))
    tmp31 = tmp29 < tmp30
    tmp32 = tmp28 & tmp31
    tmp33 = 0.0
    tmp34 = tl.where(tmp32, tmp25, tmp33)
    tmp35 = tmp23 + tmp34
    tmp36 = 0.9622504486493761
    tmp37 = tmp35 * tmp36
    tmp38 = 1.7015043497085571
    tmp39 = tmp37 * tmp38
    tmp40 = 0.7071067811865476
    tmp41 = tmp22 * tmp40
    tmp42 = tl.math.erf(tmp41)
    tmp43 = 1.0
    tmp44 = tmp42 + tmp43
    tmp45 = 0.5
    tmp46 = tmp44 * tmp45
    tmp47 = tmp22 * tmp22
    tmp48 = -0.5
    tmp49 = tmp47 * tmp48
    tmp50 = tl.exp(tmp49)
    tmp51 = 0.3989422804014327
    tmp52 = tmp50 * tmp51
    tmp53 = tmp22 * tmp52
    tmp54 = tmp46 + tmp53
    tmp55 = tmp39 * tmp54
    tmp56 = tmp55 * tmp9
    tmp57 = tmp56 * tmp7
    tmp58 = tmp57 * tmp4
    tmp59 = tmp58 * tmp0
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask, tmp60, 0)
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp62, 0))
    tmp64 = tmp43 - tmp2
    tmp65 = tmp2 * tmp64
    tmp66 = tmp63 * tmp65
    tl.store(in_out_ptr0 + (r1 + (576*x0)), tmp55, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp66, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citskd6ulb3jcnxkgu7ou5dcrymqyx4dk3jxa57wx56bqc56gzx6.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/cabo6c6eqj43aqhfoisbb34jhcbdsmdieg7ntegtqvxxva7bmnby.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 3072
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (110592*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4y/c4ye2ftxtgbc44wlf5x6ta6jbeimrg4stfxehiu7tw5f2vcqtke6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0006510416666666666
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.02551551815399144
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.02551551815399144
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co75wlgeumssyowckr7dsbderp6e3s7qxtshznw7dki6cclcrccf.py
# Source Nodes: [gelu_47, mul_101, mul_93, out_85, out_93, sigmoid_10, sigmoid_11], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
# gelu_47 => add_110, erf_47, mul_400
# mul_101 => mul_428
# mul_93 => mul_395
# out_85 => mul_396
# out_93 => mul_429
# sigmoid_10 => sigmoid_10
# sigmoid_11 => sigmoid_11
triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 54
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp44 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 36)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr5 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr6 + (((r1 + (8192*x0)) // 36)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp14 = 0.9622504486493761
        tmp15 = tmp13 * tmp14
        tmp16 = 1.7015043497085571
        tmp17 = tmp15 * tmp16
        tmp19 = 0.7071067811865476
        tmp20 = tmp18 * tmp19
        tmp21 = tl.math.erf(tmp20)
        tmp22 = 1.0
        tmp23 = tmp21 + tmp22
        tmp24 = 0.5
        tmp25 = tmp23 * tmp24
        tmp26 = tmp18 * tmp18
        tmp27 = -0.5
        tmp28 = tmp26 * tmp27
        tmp29 = tl.exp(tmp28)
        tmp30 = 0.3989422804014327
        tmp31 = tmp29 * tmp30
        tmp32 = tmp18 * tmp31
        tmp33 = tmp25 + tmp32
        tmp34 = tmp17 * tmp33
        tmp35 = tmp0 + tmp34
        tmp36 = tmp35 * tmp1
        tmp39 = tl.sigmoid(tmp38)
        tmp40 = tmp37 * tmp39
        tmp41 = tmp40 * tmp7
        tmp42 = tmp36 * tmp41
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(rmask & xmask, tmp45, _tmp44)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnituvk3wh4533arim24ycgydopvjmcbidv5czlgxabpcyat23u.py
# Source Nodes: [mul_101, out_93, sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_101 => mul_428
# out_93 => mul_429
# sigmoid_11 => sigmoid_11
triton_per_fused_mul_sigmoid_sum_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sum_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 54
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/is/ciskly7ijxyuh4errwa3hkhu22oao3jx5pphqvmfgfxwnyrykwii.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cuacpllouxykco3onu74te63vafkcua2blzpedlfjrwmd6dpkmrk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu33m4hk4acdfej3xkj5huw4jykktb4ajilwajd5rlxo6qwo4zzm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (55296*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpv37kaaxnj2nsy5cp2uf3qth746emsdrn2apmta544qtnavgir.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1536
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
    tmp5 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.0013020833333333333
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.03608439182435161
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywmwsnzvklno53ty7iztlv7yqwxdqgfrc66lklk7oa3fn7encvl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (27648*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33kqaneo7i2h4incgepzdy477dblh4xx2awstv5chjqfifiou33.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0008680555555555555
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.02946278254943948
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.02946278254943948
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5szdokkdjxkkdcbaml654b4cq4d3yvhteapuketyjof7cg6luq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0006510416666666666
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.02551551815399144
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.02551551815399144
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkf2fw2oezzre5x6nun5kjph4ne6o4j4az3canujt2j2hb2nbpu.py
# Source Nodes: [mul_85, out_77, sigmoid_9], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_85 => mul_362
# out_77 => mul_363
# sigmoid_9 => sigmoid_9
triton_red_fused_mul_sigmoid_sum_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sigmoid_sum_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 54
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 36)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlvpoannqjbrcvexm47kxoxlx77bvv52vkia345zlcolv5pchkk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 144
        r2 = (rindex // 144)
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (110592*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cja7c2plisjbkqanrdzlfq2awigu5pda4drd3w5fvewipyxuzk5y.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0006510416666666666
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.02551551815399144
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.02551551815399144
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cviin2gceg4q64vmutlmwuuwogqosyo5moqnylmmgysg4kcucexl.py
# Source Nodes: [mul_68, mul_76, out_61, out_69, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
# mul_68 => mul_293
# mul_76 => mul_326
# out_61 => mul_294
# out_69 => mul_327
# sigmoid_7 => sigmoid_7
# sigmoid_8 => sigmoid_8
triton_red_fused_add_gelu_backward_mul_sigmoid_sum_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_backward_mul_sigmoid_sum_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr5 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr6 + (((r1 + (8192*x0)) // 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp14 = 0.9128709291752768
        tmp15 = tmp13 * tmp14
        tmp16 = 1.7015043497085571
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp0 + tmp19
        tmp21 = tmp20 * tmp1
        tmp24 = tl.sigmoid(tmp23)
        tmp25 = tmp22 * tmp24
        tmp26 = tmp25 * tmp7
        tmp27 = tmp21 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycsh4o55c3ywhedskevmmhl6y7vqjh6uaotrria4vzd6iucos7q.py
# Source Nodes: [mul_76, out_69, sigmoid_8], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_76 => mul_326
# out_69 => mul_327
# sigmoid_8 => sigmoid_8
triton_per_fused_mul_sigmoid_sum_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sum_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 216
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/conmnabmrcxumtw5yuzbdqve5u5zlesjsvr32rx4rbbkcvkf25io.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 144
        r2 = (rindex // 144)
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0) + (221184*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3p2texs774e6j5exl2ea34xkadbflzws4o7t4kvwk3ktrwzi72.py
# Source Nodes: [mul_52, mul_60, out_45, out_53, sigmoid_5, sigmoid_6], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
# mul_52 => mul_227
# mul_60 => mul_260
# out_45 => mul_228
# out_53 => mul_261
# sigmoid_5 => sigmoid_5
# sigmoid_6 => sigmoid_6
triton_red_fused_add_gelu_backward_mul_sigmoid_sum_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_backward_mul_sigmoid_sum_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr5 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr6 + (((r1 + (8192*x0)) // 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp14 = 0.9449111825230679
        tmp15 = tmp13 * tmp14
        tmp16 = 1.7015043497085571
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp0 + tmp19
        tmp21 = tmp20 * tmp1
        tmp24 = tl.sigmoid(tmp23)
        tmp25 = tmp22 * tmp24
        tmp26 = tmp25 * tmp7
        tmp27 = tmp21 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqwigxzvp23gzzsd2mjwgdb5o2er3jxvn2ifgirnrkgmsb3bift.py
# Source Nodes: [mul_36, out_29, sigmoid_3], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_36 => mul_161
# out_29 => mul_162
# sigmoid_3 => sigmoid_3
triton_red_fused_mul_sigmoid_sum_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sigmoid_sum_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 144)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw756smqsmflnnrkvsufmzqa3pefeu7rwhxsdbwggfukz6m3whz2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 576
        r2 = (rindex // 576)
        tmp0 = tl.load(in_ptr0 + (r1 + (576*x0) + (442368*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2n5gaw7facr4rjnfb6fqbrmiwl4gbshmfjpddemg3lxvsfx2qk.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 768
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.04419417382415922
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmyrriviar6ho6sjz644l6y62rfhdjnckg6f6fsvwpf2v4nku4tl.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1536
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.04419417382415922
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/ccl3gszxsxuwgk2xhwwww2p6wze4tg2zjnvaaj4zxowrimhow423.py
# Source Nodes: [mul_27, out_21, sigmoid_2], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_27 => mul_125
# out_21 => mul_126
# sigmoid_2 => sigmoid_2
triton_red_fused_mul_sigmoid_sum_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sigmoid_sum_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 576)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccylcthuzktyhwnjbieog4s7ebrrk27zptjndg4pahvsd35ouvuf.py
# Source Nodes: [mul_27, out_21, sigmoid_2], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_27 => mul_125
# out_21 => mul_126
# sigmoid_2 => sigmoid_2
triton_per_fused_mul_sigmoid_sum_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sum_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tl/ctl3ani36ppfkwekeredqpm3he5wxo5dluctcwv6liemqkik6sku.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/vw/cvw73agbkjkbfs65p7z5osqwgrcoaaxtckfj6za4rvzyvpbe3qr2.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56igiyezu2eegotcdo5fdb2yjc5bs7yfwousucaurcuol2tshns.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sx/csxzvsvw5hiqdsfkv2udl7gwkulmvs4ristxpeej3k4hww67crpu.py
# Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid_2 => sigmoid_2
triton_poi_fused_add_div_mul_sigmoid_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 576)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp7 * tmp9
    tmp12 = 576.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dp/cdperxtetc43qrh4xzv57adqlaha6nhbtcyjm3gwcvybqieijz4a.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 576
        r2 = (rindex // 576)
        tmp0 = tl.load(in_ptr0 + (r1 + (576*x0) + (294912*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bi/cbipt7sj7ztaxbztoz43zgxc7wsazi6de5aycp4wnwzqknkpyezb.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.00390625
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.0625
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxvuls2gejignsiseoiq5hrjpejyej3zp7k7ti2fjrmkofanpsw.py
# Source Nodes: [gelu_14], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
# gelu_14 => add_33, erf_14, mul_119
triton_poi_fused_gelu_gelu_backward_mul_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_mul_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.7015043497085571
    tmp2 = tmp0 * tmp1
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp2 * tmp18
    tl.store(in_out_ptr0 + (x0), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caoutgrouec73lejgzdx4ukhkrgnpe6xcop7ds6b6dronhacbm2c.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 576
        r2 = (rindex // 576)
        tmp0 = tl.load(in_ptr0 + (r1 + (576*x0) + (147456*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzrmelufixonfz2z7s2nbaxa3ge2g232vl4ednftpz6gakde7bs.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0008680555555555555
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.02946278254943948
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.02946278254943948
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camqo26a4mgfg2fqrmz4b5h4jaa5x26ic3j2qwmmu2qf2gmhdint.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 256
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.04419417382415922
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskcrtvpjv2mnbbunpyzn4t4stcznjgrxu6bpxlvhlxngb5hatuy.py
# Source Nodes: [gelu_11, mul_19, mul_21, mul__12, out_13, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_11 => add_27, erf_11, mul_97
# mul_19 => mul_92
# mul_21 => mul_95
# mul__12 => mul_94
# out_13 => mul_93
# shortcut_4 => add_26
# sigmoid_1 => sigmoid_1
triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_57', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (576*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_out_ptr0 + (r1 + (576*x0)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr4 + (r1 + (576*x0)), rmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 0.7071067811865476
    tmp14 = tmp12 * tmp13
    tmp15 = tl.math.erf(tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.5
    tmp19 = tmp17 * tmp18
    tmp20 = tmp12 * tmp12
    tmp21 = -0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tl.exp(tmp22)
    tmp24 = 0.3989422804014327
    tmp25 = tmp23 * tmp24
    tmp26 = tmp12 * tmp25
    tmp27 = tmp19 + tmp26
    tmp30 = 0.9805806756909201
    tmp31 = tmp29 * tmp30
    tmp32 = 1.7015043497085571
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33 * tmp27
    tmp35 = tmp28 + tmp34
    tmp36 = tmp35 * tmp9
    tmp37 = tmp36 * tmp7
    tmp38 = tmp37 * tmp4
    tmp39 = tmp38 * tmp0
    tmp40 = tl.broadcast_to(tmp39, [RBLOCK])
    tmp42 = tl.where(rmask, tmp40, 0)
    tmp43 = triton_helpers.promote_to_tensor(tl.sum(tmp42, 0))
    tmp44 = tmp16 - tmp2
    tmp45 = tmp2 * tmp44
    tmp46 = tmp43 * tmp45
    tl.store(in_out_ptr0 + (r1 + (576*x0)), tmp35, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/canmentia5as6yc3wkxfqs2g6r7gxoizoimgjrzzvelf66fkuvuj.py
# Source Nodes: [gelu_8], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
# gelu_8 => add_20, erf_8, mul_72
triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 48) % 48
    x0 = xindex % 48
    x2 = (xindex // 2304)
    x3 = xindex
    tmp11 = tl.load(in_ptr1 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 49, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (49*x1) + (2401*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1.7015043497085571
    tmp10 = tmp8 * tmp9
    tmp12 = 0.7071067811865476
    tmp13 = tmp11 * tmp12
    tmp14 = tl.math.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tmp11 * tmp11
    tmp20 = -0.5
    tmp21 = tmp19 * tmp20
    tmp22 = tl.exp(tmp21)
    tmp23 = 0.3989422804014327
    tmp24 = tmp22 * tmp23
    tmp25 = tmp11 * tmp24
    tmp26 = tmp18 + tmp25
    tmp27 = tmp10 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwa4purvoe4ckjhmmhpgdmky3af6oxpq4qdbygjdccz2cglfgdt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 2304
        r2 = (rindex // 2304)
        tmp0 = tl.load(in_ptr0 + (r1 + (2304*x0) + (589824*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmqjvmpi7bp7mcxjrwgnznifzdrafix5ittbn63jgyzgooqbmvo.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.00390625
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.0625
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5rqfrqpqx2tyusgsjzevrrsr4lhzezywhqmube52rzyk2n2mfi.py
# Source Nodes: [gelu_7, mul_10, mul_12, mul__7, out_5, shortcut_2, sigmoid], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# gelu_7 => add_17, erf_7, mul_61
# mul_10 => mul_56
# mul_12 => mul_59
# mul__7 => mul_58
# out_5 => mul_57
# shortcut_2 => add_16
# sigmoid => sigmoid
triton_red_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_61', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp51 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        r2 = rindex % 48
        r3 = (rindex // 48)
        tmp0 = tl.load(in_ptr0 + (r1 + (2304*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (2304*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_out_ptr0 + (r1 + (2304*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr4 + ((24*(tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(24, 1 + (r3 // 2)))))) + (24*(tl.where((tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(24, 1 + (r3 // 2))))) >= 0, 0, 24))) + (576*x0) + (tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(24, 1 + (r2 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(24, 1 + (r2 // 2))))) >= 0, 0, 24))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp8 = tmp5 * tmp7
        tmp9 = 0.2
        tmp10 = tmp8 * tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = 0.7071067811865476
        tmp14 = tmp12 * tmp13
        tmp15 = tl.math.erf(tmp14)
        tmp16 = 1.0
        tmp17 = tmp15 + tmp16
        tmp18 = 0.5
        tmp19 = tmp17 * tmp18
        tmp20 = tmp12 * tmp12
        tmp21 = -0.5
        tmp22 = tmp20 * tmp21
        tmp23 = tl.exp(tmp22)
        tmp24 = 0.3989422804014327
        tmp25 = tmp23 * tmp24
        tmp26 = tmp12 * tmp25
        tmp27 = tmp19 + tmp26
        tmp30 = tmp29 / 4
        tmp31 = tl.math.max(0, (r3 // 2))
        tmp32 = tl.math.min(24, 1 + (r3 // 2))
        tmp33 = tmp31 < tmp32
        tmp34 = tl.math.max(0, (r2 // 2))
        tmp35 = tl.math.min(24, 1 + (r2 // 2))
        tmp36 = tmp34 < tmp35
        tmp37 = tmp33 & tmp36
        tmp38 = 0.0
        tmp39 = tl.where(tmp37, tmp30, tmp38)
        tmp40 = tmp28 + tmp39
        tmp41 = 0.9805806756909201
        tmp42 = tmp40 * tmp41
        tmp43 = 1.7015043497085571
        tmp44 = tmp42 * tmp43
        tmp45 = tmp44 * tmp27
        tmp46 = tmp45 * tmp9
        tmp47 = tmp46 * tmp7
        tmp48 = tmp47 * tmp4
        tmp49 = tmp48 * tmp0
        tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
        tmp52 = _tmp51 + tmp50
        _tmp51 = tl.where(rmask, tmp52, _tmp51)
        tl.store(in_out_ptr0 + (r1 + (2304*x0)), tmp45, rmask)
    tmp51 = tl.sum(_tmp51, 1)[:, None]
    tmp53 = tl.sigmoid(tmp1)
    tmp54 = 1.0
    tmp55 = tmp54 - tmp53
    tmp56 = tmp53 * tmp55
    tmp57 = tmp51 * tmp56
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dxfqyvrpmw4fficiyh4z5m5qj2nsy2ekieuyjjunbki233vxgs.py
# Source Nodes: [mul_10, out_5, sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_10 => mul_56
# out_5 => mul_57
# sigmoid => sigmoid
triton_red_fused_mul_sigmoid_sum_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sigmoid_sum_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (((r1 + (8192*x0)) // 2304)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.2
        tmp2 = tmp0 * tmp1
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 2.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshbt67f5gmmd7is6n4pdrt6lpm5l4pxxgarbrtyjgagxk65atzi.py
# Source Nodes: [mul_10, out_5, sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
# mul_10 => mul_56
# out_5 => mul_57
# sigmoid => sigmoid
triton_per_fused_mul_sigmoid_sum_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sum_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmajfgzwpej2vcvflyxibgm6jnxnp4mb5x7ymvrabxhrn2c5lt6n.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7n/c7nrasdg7psb7ttwtl3dmactenav2xdozowpssv2fqqw7cnmvkzg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3qyamp4vbl2u3vg7uj5jkjny5h7ucncnuwdz2kj2itamo3t7ic.py
# Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
# sigmoid => sigmoid
triton_poi_fused_add_div_mul_sigmoid_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sigmoid_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2304)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.2
    tmp2 = tmp0 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp7 * tmp9
    tmp12 = 2304.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pwbhflgz32wn4bbibol2sdgec5msor2zhw4xaj7ylga4qckbph.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp5 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.0078125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.08838834764831845
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2zpsznsaghyoai2yil3jdmt5xowxhddttdf36p5ooekh6tzsmx.py
# Source Nodes: [gelu_6], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
# gelu_6 => add_14, erf_6, mul_50
triton_poi_fused_gelu_gelu_backward_mul_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_mul_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 1.7015043497085571
    tmp2 = tmp0 * tmp1
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp2 * tmp18
    tl.store(in_out_ptr0 + (x0), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5qlrz3lxus2i3jyzytiwsfxyqsci2h5fbeyhklmqlzz6xmpd3n.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((48*(((r2 + (6144*x1)) // 48) % 48)) + (2304*x0) + (294912*((r2 + (6144*x1)) // 2304)) + (r2 % 48)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdywgf2qqvuwpcd5633bn643iwxwqdgz4t7uovdtf64diysxg3yx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgesynffo5ttzmofr6gntwqo53ybrxjnqyjncufxsrtvgakfm7fq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_red_fused_mul_native_batch_norm_backward_view_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_view_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12 - tmp5
        tmp14 = 0.0008680555555555555
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp15 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tmp11 - tmp19
        tmp21 = tmp2 * tmp14
        tmp22 = tmp20 - tmp21
        tmp24 = 0.02946278254943948
        tmp25 = tmp23 * tmp24
        tmp26 = tmp16 * tmp25
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp27, rmask & xmask)
    tmp28 = tmp9 * tmp16
    tmp29 = 0.02946278254943948
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cua4burbxxa5hqyk4wvltvclrgfqijltdvrcbs3goilw22c3zp3t.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp5 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.0078125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.08838834764831845
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cflckkspmp3tkhs3bo7gr5sdi7sxsc6v6bq4727rb455fxxy3xbt.py
# Source Nodes: [gelu_3], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul]
# gelu_3 => add_7, erf_3, mul_25
triton_poi_fused_add_gelu_gelu_backward_mul_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_gelu_backward_mul_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 1.7015043497085571
    tmp6 = tmp4 * tmp5
    tmp8 = 0.7071067811865476
    tmp9 = tmp7 * tmp8
    tmp10 = tl.math.erf(tmp9)
    tmp11 = tmp10 + tmp3
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp7 * tmp7
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tl.exp(tmp16)
    tmp18 = 0.3989422804014327
    tmp19 = tmp17 * tmp18
    tmp20 = tmp7 * tmp19
    tmp21 = tmp13 + tmp20
    tmp22 = tmp6 * tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtytedcdnbhfmeqtz7bcbtzjsi5ilqdzjjpwiwjqijhfmn5p3w6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.001736111111111111
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.041666666666666664
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zwh4rlqa5d673kw3jbfq2t6m6m7imwinqv4xge6vdf7h5lvqpw.py
# Source Nodes: [gelu_2], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
# gelu_2 => add_5, erf_2, mul_18
triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 96) % 96
    x0 = xindex % 96
    x2 = (xindex // 9216)
    x3 = xindex
    tmp11 = tl.load(in_ptr1 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 97, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (97*x1) + (9409*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1.7015043497085571
    tmp10 = tmp8 * tmp9
    tmp12 = 0.7071067811865476
    tmp13 = tmp11 * tmp12
    tmp14 = tl.math.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tmp11 * tmp11
    tmp20 = -0.5
    tmp21 = tmp19 * tmp20
    tmp22 = tl.exp(tmp21)
    tmp23 = 0.3989422804014327
    tmp24 = tmp22 * tmp23
    tmp25 = tmp11 * tmp24
    tmp26 = tmp18 + tmp25
    tmp27 = tmp10 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7kmousha3d56jfm7witgeet3lxew7ncinvfh4xka2phovpuxsu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (589824*((r2 + (8192*x0)) // 9216)) + ((r2 + (8192*x0)) % 9216)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ye/cyeecuqn7jvtpz4fckwsds7u2aak44pbtuw4qj47tok4rf6iadgu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coq532sag43dleuif5xvs4akmbly5cont6ot3rirmnxpajg53jzq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (288*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (288*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 0.003472222222222222
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.05892556509887896
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (288*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyezfi6w7jcsenndmdixiqtlrajr74776uezjfy7mtgcme5rjwa.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (294912*((r2 + (8192*x0)) // 9216)) + ((r2 + (8192*x0)) % 9216)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqtkuoi3z7ep6neao34u5e3gzfqmmlcf4b4mzpvfsuuctz7wwu4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqxdpsgyihyap3ryknzjptg5z2cabrrgwkqy3w7wvzrobeeuanm.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.006944444444444444
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.08333333333333333
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c277oee4mkbdeqbidloubaw2gi5hqawukyt7df7yxkzmxfcnq4rs.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 9
    x1 = (xindex // 9)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((9216*x1) + (147456*((r2 + (8192*x0)) // 9216)) + ((r2 + (8192*x0)) % 9216)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2edbfyaizpkw7i24te2bimeume6hm4nsclndcae7fnfukho73xl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpn42mowci7h22qy22uywdw6mtbrikwlube4kocqjcdxvqw7svx3.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]

triton_per_fused_mul_native_batch_norm_backward_view_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_view_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 27
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (27*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (27*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 0.037037037037037035
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp7 * tmp17
    tmp19 = tmp0 - tmp18
    tmp20 = tmp4 * tmp13
    tmp21 = tmp19 - tmp20
    tmp23 = 0.19245008972987526
    tmp24 = tmp22 * tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp12 * tmp15
    tmp28 = tmp27 * tmp23
    tl.store(out_ptr2 + (r1 + (27*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_83, primals_84, primals_86, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_156, primals_158, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, constant_pad_nd, squeeze_1, view_2, convolution, mul_6, squeeze_3, view_5, convolution_1, mul_13, squeeze_5, view_8, convolution_2, constant_pad_nd_1, squeeze_7, view_11, convolution_3, mul_28, squeeze_9, view_14, convolution_4, squeeze_11, view_17, convolution_5, mul_38, squeeze_13, view_20, convolution_6, mul_45, squeeze_15, view_23, convolution_7, mul_52, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_64, avg_pool2d, squeeze_19, view_29, convolution_11, squeeze_21, view_32, convolution_12, constant_pad_nd_2, squeeze_23, view_35, convolution_13, mul_81, squeeze_25, view_38, convolution_14, mul_88, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_100, squeeze_29, view_44, convolution_18, mul_107, squeeze_31, view_47, convolution_19, mul_114, squeeze_33, view_50, convolution_20, mul_121, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_133, avg_pool2d_1, squeeze_37, view_56, convolution_24, squeeze_39, view_59, convolution_25, constant_pad_nd_3, squeeze_41, view_62, convolution_26, mul_150, squeeze_43, view_65, convolution_27, mul_157, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_169, squeeze_47, view_71, convolution_31, mul_176, squeeze_49, view_74, convolution_32, mul_183, squeeze_51, view_77, convolution_33, mul_190, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_202, squeeze_55, view_83, convolution_37, mul_209, squeeze_57, view_86, convolution_38, mul_216, squeeze_59, view_89, convolution_39, mul_223, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_235, squeeze_63, view_95, convolution_43, mul_242, squeeze_65, view_98, convolution_44, mul_249, squeeze_67, view_101, convolution_45, mul_256, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_268, squeeze_71, view_107, convolution_49, mul_275, squeeze_73, view_110, convolution_50, mul_282, squeeze_75, view_113, convolution_51, mul_289, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_301, squeeze_79, view_119, convolution_55, mul_308, squeeze_81, view_122, convolution_56, mul_315, squeeze_83, view_125, convolution_57, mul_322, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_334, avg_pool2d_2, squeeze_87, view_131, convolution_61, squeeze_89, view_134, convolution_62, constant_pad_nd_4, squeeze_91, view_137, convolution_63, mul_351, squeeze_93, view_140, convolution_64, mul_358, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_370, squeeze_97, view_146, convolution_68, mul_377, squeeze_99, view_149, convolution_69, mul_384, squeeze_101, view_152, convolution_70, mul_391, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_403, squeeze_105, view_158, convolution_74, mul_410, squeeze_107, view_161, convolution_75, mul_417, squeeze_109, view_164, convolution_76, mul_424, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_118, squeeze_113, view_170, convolution_80, clone_12, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_4, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_5, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_10, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_13, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_14, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_16, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_19, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_20, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_22, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_23, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_25, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_26, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_28, (), ())
    assert_size_stride(primals_29, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_30, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_32, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_33, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_35, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_36, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_38, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_39, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_41, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_42, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_44, (), ())
    assert_size_stride(primals_45, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_46, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_48, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_49, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_51, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_52, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_54, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_55, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_57, (), ())
    assert_size_stride(primals_58, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_61, (768, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_62, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_64, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_65, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_67, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_68, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_70, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_71, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_73, (), ())
    assert_size_stride(primals_74, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_75, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_77, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_78, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_80, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_81, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_83, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_84, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_86, (), ())
    assert_size_stride(primals_87, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_88, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_90, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_91, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_93, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_94, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_96, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_97, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_99, (), ())
    assert_size_stride(primals_100, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_101, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_103, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_104, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_106, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_107, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_109, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_110, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_112, (), ())
    assert_size_stride(primals_113, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_114, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_116, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_117, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_119, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_120, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_122, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_123, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_125, (), ())
    assert_size_stride(primals_126, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_127, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_129, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_130, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_132, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_133, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_135, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_136, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_138, (), ())
    assert_size_stride(primals_139, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_140, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_142, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_143, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_145, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_146, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_148, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_149, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_151, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_152, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_154, (), ())
    assert_size_stride(primals_155, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_156, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_158, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_159, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_161, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_162, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_164, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_165, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_169, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_171, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_172, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_174, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_175, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_177, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_178, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (3072, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_182, (3072, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_184, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_186, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_188, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_190, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_192, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_194, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_196, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_198, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_200, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_202, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_204, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_206, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_208, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_210, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_212, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_214, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_216, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_218, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_220, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_222, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_224, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_226, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_228, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_230, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(constant_pad_nd, (8, 3, 193, 193), (111747, 37249, 193, 1))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(view_2, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(convolution, (8, 16, 96, 96), (147456, 9216, 96, 1))
    assert_size_stride(mul_6, (8, 16, 96, 96), (147456, 9216, 96, 1))
    assert_size_stride(squeeze_3, (32, ), (1, ))
    assert_size_stride(view_5, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(convolution_1, (8, 32, 96, 96), (294912, 9216, 96, 1))
    assert_size_stride(mul_13, (8, 32, 96, 96), (294912, 9216, 96, 1))
    assert_size_stride(squeeze_5, (64, ), (1, ))
    assert_size_stride(view_8, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(convolution_2, (8, 64, 96, 96), (589824, 9216, 96, 1))
    assert_size_stride(constant_pad_nd_1, (8, 64, 97, 97), (602176, 9409, 97, 1))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(view_11, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(convolution_3, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(mul_28, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(squeeze_9, (256, ), (1, ))
    assert_size_stride(view_14, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_4, (8, 256, 48, 48), (589824, 2304, 48, 1))
    assert_size_stride(squeeze_11, (128, ), (1, ))
    assert_size_stride(view_17, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_5, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(mul_38, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(squeeze_13, (128, ), (1, ))
    assert_size_stride(view_20, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_6, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(mul_45, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(squeeze_15, (128, ), (1, ))
    assert_size_stride(view_23, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_7, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(mul_52, (8, 128, 48, 48), (294912, 2304, 48, 1))
    assert_size_stride(squeeze_17, (256, ), (1, ))
    assert_size_stride(view_26, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_8, (8, 256, 48, 48), (589824, 2304, 48, 1))
    assert_size_stride(mean, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(relu, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_10, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_64, (8, 256, 48, 48), (589824, 2304, 48, 1))
    assert_size_stride(avg_pool2d, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(squeeze_19, (512, ), (1, ))
    assert_size_stride(view_29, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_11, (8, 512, 24, 24), (294912, 576, 24, 1))
    assert_size_stride(squeeze_21, (256, ), (1, ))
    assert_size_stride(view_32, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_12, (8, 256, 48, 48), (589824, 2304, 48, 1))
    assert_size_stride(constant_pad_nd_2, (8, 256, 49, 49), (614656, 2401, 49, 1))
    assert_size_stride(squeeze_23, (256, ), (1, ))
    assert_size_stride(view_35, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_13, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(mul_81, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(squeeze_25, (256, ), (1, ))
    assert_size_stride(view_38, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_14, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(mul_88, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(squeeze_27, (512, ), (1, ))
    assert_size_stride(view_41, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_15, (8, 512, 24, 24), (294912, 576, 24, 1))
    assert_size_stride(mean_1, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(relu_1, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_17, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_100, (8, 512, 24, 24), (294912, 576, 24, 1))
    assert_size_stride(squeeze_29, (256, ), (1, ))
    assert_size_stride(view_44, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_18, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(mul_107, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(view_47, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_19, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(mul_114, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(squeeze_33, (256, ), (1, ))
    assert_size_stride(view_50, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_20, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(mul_121, (8, 256, 24, 24), (147456, 576, 24, 1))
    assert_size_stride(squeeze_35, (512, ), (1, ))
    assert_size_stride(view_53, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_21, (8, 512, 24, 24), (294912, 576, 24, 1))
    assert_size_stride(mean_2, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(relu_2, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_23, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_133, (8, 512, 24, 24), (294912, 576, 24, 1))
    assert_size_stride(avg_pool2d_1, (8, 512, 12, 12), (73728, 144, 12, 1))
    assert_size_stride(squeeze_37, (1536, ), (1, ))
    assert_size_stride(view_56, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_24, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(squeeze_39, (768, ), (1, ))
    assert_size_stride(view_59, (768, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_25, (8, 768, 24, 24), (442368, 576, 24, 1))
    assert_size_stride(constant_pad_nd_3, (8, 768, 25, 25), (480000, 625, 25, 1))
    assert_size_stride(squeeze_41, (768, ), (1, ))
    assert_size_stride(view_62, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_26, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_150, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_43, (768, ), (1, ))
    assert_size_stride(view_65, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_27, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_157, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_45, (1536, ), (1, ))
    assert_size_stride(view_68, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_28, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(mean_3, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_3, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_30, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_169, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(squeeze_47, (768, ), (1, ))
    assert_size_stride(view_71, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_31, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_176, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_49, (768, ), (1, ))
    assert_size_stride(view_74, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_32, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_183, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_51, (768, ), (1, ))
    assert_size_stride(view_77, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_33, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_190, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_53, (1536, ), (1, ))
    assert_size_stride(view_80, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_34, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(mean_4, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_4, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_36, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_202, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(squeeze_55, (768, ), (1, ))
    assert_size_stride(view_83, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_37, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_209, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_57, (768, ), (1, ))
    assert_size_stride(view_86, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_38, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_216, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_59, (768, ), (1, ))
    assert_size_stride(view_89, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_39, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_223, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_61, (1536, ), (1, ))
    assert_size_stride(view_92, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_40, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(mean_5, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_5, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_42, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_235, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(squeeze_63, (768, ), (1, ))
    assert_size_stride(view_95, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_43, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_242, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_65, (768, ), (1, ))
    assert_size_stride(view_98, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_44, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_249, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_67, (768, ), (1, ))
    assert_size_stride(view_101, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_45, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_256, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_69, (1536, ), (1, ))
    assert_size_stride(view_104, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_46, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(mean_6, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_6, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_48, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_268, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(squeeze_71, (768, ), (1, ))
    assert_size_stride(view_107, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_49, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_275, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_73, (768, ), (1, ))
    assert_size_stride(view_110, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_50, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_282, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_75, (768, ), (1, ))
    assert_size_stride(view_113, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_51, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_289, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_77, (1536, ), (1, ))
    assert_size_stride(view_116, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_52, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(mean_7, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_7, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_54, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_301, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(squeeze_79, (768, ), (1, ))
    assert_size_stride(view_119, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_55, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_308, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_81, (768, ), (1, ))
    assert_size_stride(view_122, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_56, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_315, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_83, (768, ), (1, ))
    assert_size_stride(view_125, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_57, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(mul_322, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(squeeze_85, (1536, ), (1, ))
    assert_size_stride(view_128, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_58, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(mean_8, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_8, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_60, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_334, (8, 1536, 12, 12), (221184, 144, 12, 1))
    assert_size_stride(avg_pool2d_2, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(squeeze_87, (1536, ), (1, ))
    assert_size_stride(view_131, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_61, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(squeeze_89, (768, ), (1, ))
    assert_size_stride(view_134, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_62, (8, 768, 12, 12), (110592, 144, 12, 1))
    assert_size_stride(constant_pad_nd_4, (8, 768, 13, 13), (129792, 169, 13, 1))
    assert_size_stride(squeeze_91, (768, ), (1, ))
    assert_size_stride(view_137, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_63, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_351, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_93, (768, ), (1, ))
    assert_size_stride(view_140, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_64, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_358, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_95, (1536, ), (1, ))
    assert_size_stride(view_143, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_65, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(mean_9, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_9, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_67, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_370, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(squeeze_97, (768, ), (1, ))
    assert_size_stride(view_146, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_68, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_377, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_99, (768, ), (1, ))
    assert_size_stride(view_149, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_69, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_384, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_101, (768, ), (1, ))
    assert_size_stride(view_152, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_70, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_391, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_103, (1536, ), (1, ))
    assert_size_stride(view_155, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_71, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(mean_10, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_10, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_73, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_403, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(squeeze_105, (768, ), (1, ))
    assert_size_stride(view_158, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_74, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_410, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_107, (768, ), (1, ))
    assert_size_stride(view_161, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_75, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_417, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_109, (768, ), (1, ))
    assert_size_stride(view_164, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(convolution_76, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(mul_424, (8, 768, 6, 6), (27648, 36, 6, 1))
    assert_size_stride(squeeze_111, (1536, ), (1, ))
    assert_size_stride(view_167, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_77, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(mean_11, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(relu_11, (8, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(convolution_79, (8, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(add_118, (8, 1536, 6, 6), (55296, 36, 6, 1))
    assert_size_stride(squeeze_113, (3072, ), (1, ))
    assert_size_stride(view_170, (3072, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(convolution_80, (8, 3072, 6, 6), (110592, 36, 6, 1))
    assert_size_stride(clone_12, (8, 3072), (3072, 1))
    assert_size_stride(permute_1, (1000, 3072), (3072, 1))
    assert_size_stride(unsqueeze_58, (1, 3072, 1), (3072, 1, 1))
    assert_size_stride(unsqueeze_66, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_74, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_82, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_90, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_98, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_106, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_114, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_122, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_130, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_138, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_146, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_154, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_162, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_170, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_186, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_194, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_210, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_218, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_234, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_266, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 32, 1), (32, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 16, 1), (16, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf5 = empty((8, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf5)
        del permute_1
        buf8 = empty((8, 3072, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_51], Original ATen: [aten.div, aten.gelu, aten.gelu_backward, aten.mul]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_div_gelu_gelu_backward_mul_0.run(buf5, convolution_80, buf8, 884736, grid=grid(884736), stream=stream0)
        del buf5
        del convolution_80
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf10 = aten.convolution_backward(buf8, add_118, view_170, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_118
        del view_170
        buf11 = buf10[0]
        buf19 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf20 = reinterpret_tensor(buf19, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf19  # reuse
        # Source Nodes: [sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_sum_1.run(buf20, buf11, primals_180, convolution_77, convolution_79, 12288, 36, grid=grid(12288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf22 = aten.convolution_backward(buf20, relu_11, primals_230, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_230
        buf23 = buf22[0]
        buf25 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf25, relu_11, 6144, grid=grid(6144), stream=stream0)
        del relu_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf27 = aten.convolution_backward(buf25, mean_11, primals_228, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_11
        del primals_228
        buf28 = buf27[0]
        buf30 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_3.run(buf11, primals_180, convolution_79, buf28, buf30, 442368, grid=grid(442368), stream=stream0)
        del primals_180
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf32 = aten.convolution_backward(buf30, mul_424, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_424
        del view_167
        buf33 = buf32[0]
        buf39 = buf33; del buf33  # reuse
        # Source Nodes: [gelu_50], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf39, convolution_76, 221184, grid=grid(221184), stream=stream0)
        del convolution_76
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf41 = aten.convolution_backward(buf39, mul_417, view_164, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_417
        del view_164
        buf42 = buf41[0]
        buf48 = buf42; del buf42  # reuse
        # Source Nodes: [gelu_49], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf48, convolution_75, 221184, grid=grid(221184), stream=stream0)
        del convolution_75
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf50 = aten.convolution_backward(buf48, mul_410, view_161, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_410
        del view_161
        buf51 = buf50[0]
        buf57 = buf51; del buf51  # reuse
        # Source Nodes: [gelu_48], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf57, convolution_74, 221184, grid=grid(221184), stream=stream0)
        del convolution_74
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf59 = aten.convolution_backward(buf57, mul_403, view_158, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_403
        del view_158
        buf60 = buf59[0]
        buf4 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        buf68 = reinterpret_tensor(buf28, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf28  # reuse
        buf69 = reinterpret_tensor(buf68, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf68  # reuse
        # Source Nodes: [gelu_47, mul_85, mul_87, mul_93, mul_95, mul__52, mul__57, out_77, out_85, shortcut_14, shortcut_15, sigmoid_10, sigmoid_9], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_5.run(buf69, convolution_71, convolution_73, primals_167, convolution_65, convolution_67, primals_154, convolution_61, buf11, buf60, buf4, 12288, 36, grid=grid(12288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf71 = aten.convolution_backward(buf69, relu_10, primals_226, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_226
        buf72 = buf71[0]
        buf74 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf74, relu_10, 6144, grid=grid(6144), stream=stream0)
        del relu_10
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf76 = aten.convolution_backward(buf74, mean_10, primals_224, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_10
        del primals_224
        buf77 = buf76[0]
        buf79 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_47, sigmoid_10], Original ATen: [aten.add, aten.div, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_gelu_gelu_backward_mul_sigmoid_6.run(buf11, buf60, buf4, primals_167, convolution_73, buf77, buf79, 442368, grid=grid(442368), stream=stream0)
        del primals_167
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf81 = aten.convolution_backward(buf79, mul_391, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_391
        del view_155
        buf82 = buf81[0]
        buf88 = buf82; del buf82  # reuse
        # Source Nodes: [gelu_46], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf88, convolution_70, 221184, grid=grid(221184), stream=stream0)
        del convolution_70
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf90 = aten.convolution_backward(buf88, mul_384, view_152, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_384
        del view_152
        buf91 = buf90[0]
        buf97 = buf91; del buf91  # reuse
        # Source Nodes: [gelu_45], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf97, convolution_69, 221184, grid=grid(221184), stream=stream0)
        del convolution_69
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf97, mul_377, view_149, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_377
        del view_149
        buf100 = buf99[0]
        buf106 = buf100; del buf100  # reuse
        # Source Nodes: [gelu_44], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf106, convolution_68, 221184, grid=grid(221184), stream=stream0)
        del convolution_68
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf108 = aten.convolution_backward(buf106, mul_370, view_146, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_370
        del view_146
        buf109 = buf108[0]
        buf116 = buf109; del buf109  # reuse
        buf119 = reinterpret_tensor(buf77, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf77  # reuse
        buf120 = reinterpret_tensor(buf119, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf119  # reuse
        # Source Nodes: [gelu_43, gelu_47, mul_85, mul_87, mul__52, out_77, shortcut_14, sigmoid_9], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_7.run(buf116, buf120, convolution_65, convolution_67, primals_154, convolution_61, buf11, buf60, buf4, 12288, 36, grid=grid(12288), stream=stream0)
        del convolution_61
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf122 = aten.convolution_backward(buf120, relu_9, primals_222, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_222
        buf123 = buf122[0]
        buf125 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf125, relu_9, 6144, grid=grid(6144), stream=stream0)
        del relu_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf127 = aten.convolution_backward(buf125, mean_9, primals_220, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_9
        del primals_220
        buf128 = buf127[0]
        buf130 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_3.run(buf116, primals_154, convolution_67, buf128, buf130, 442368, grid=grid(442368), stream=stream0)
        del primals_154
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf132 = aten.convolution_backward(buf130, mul_358, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_358
        del view_143
        buf133 = buf132[0]
        buf139 = buf133; del buf133  # reuse
        # Source Nodes: [gelu_42], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf139, convolution_64, 221184, grid=grid(221184), stream=stream0)
        del convolution_64
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf141 = aten.convolution_backward(buf139, mul_351, view_140, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_351
        del view_140
        buf142 = buf141[0]
        buf148 = buf142; del buf142  # reuse
        # Source Nodes: [gelu_41], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_4.run(buf148, convolution_63, 221184, grid=grid(221184), stream=stream0)
        del convolution_63
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf150 = aten.convolution_backward(buf148, constant_pad_nd_4, view_137, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, False])
        del constant_pad_nd_4
        del view_137
        buf151 = buf150[0]
        buf157 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_40], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_8.run(buf151, convolution_62, buf157, 884736, grid=grid(884736), stream=stream0)
        del buf151
        del convolution_62
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf159 = aten.convolution_backward(buf157, mul_334, view_134, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_334
        del view_134
        buf160 = buf159[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf167 = aten.convolution_backward(buf116, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_2
        del view_131
        buf168 = buf167[0]
        buf1 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        buf2 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        buf174 = buf160; del buf160  # reuse
        buf224 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        buf324 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        buf177 = reinterpret_tensor(buf128, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf128  # reuse
        buf178 = reinterpret_tensor(buf177, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf177  # reuse
        # Source Nodes: [gelu_27, gelu_35, gelu_39, mul_36, mul_38, mul_44, mul_46, mul_52, mul_54, mul_60, mul_62, mul_68, mul_70, mul_76, mul_78, mul__22, mul__27, mul__32, mul__37, mul__42, mul__47, out_29, out_37, out_45, out_53, out_61, out_69, shortcut_10, shortcut_11, shortcut_12, shortcut_7, shortcut_8, shortcut_9, sigmoid_3, sigmoid_4, sigmoid_5, sigmoid_6, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_9.run(buf174, buf178, convolution_34, convolution_36, primals_86, convolution_28, convolution_30, primals_73, convolution_24, convolution_46, convolution_48, primals_112, convolution_40, convolution_42, primals_99, convolution_58, convolution_60, primals_138, convolution_52, convolution_54, primals_125, buf168, buf1, buf2, buf224, buf324, 12288, 144, grid=grid(12288), stream=stream0)
        del buf168
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf180 = aten.convolution_backward(buf178, relu_8, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_218
        buf181 = buf180[0]
        buf183 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf183, relu_8, 6144, grid=grid(6144), stream=stream0)
        del relu_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf185 = aten.convolution_backward(buf183, mean_8, primals_216, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_8
        del primals_216
        buf186 = buf185[0]
        buf188 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_10.run(buf174, primals_138, convolution_60, buf186, buf188, 1769472, grid=grid(1769472), stream=stream0)
        del primals_138
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf190 = aten.convolution_backward(buf188, mul_322, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_322
        del view_128
        buf191 = buf190[0]
        buf197 = buf191; del buf191  # reuse
        # Source Nodes: [gelu_38], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf197, convolution_57, 884736, grid=grid(884736), stream=stream0)
        del convolution_57
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf199 = aten.convolution_backward(buf197, mul_315, view_125, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_315
        del view_125
        buf200 = buf199[0]
        buf206 = buf200; del buf200  # reuse
        # Source Nodes: [gelu_37], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf206, convolution_56, 884736, grid=grid(884736), stream=stream0)
        del convolution_56
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf208 = aten.convolution_backward(buf206, mul_308, view_122, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_308
        del view_122
        buf209 = buf208[0]
        buf215 = buf209; del buf209  # reuse
        # Source Nodes: [gelu_36], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf215, convolution_55, 884736, grid=grid(884736), stream=stream0)
        del convolution_55
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf217 = aten.convolution_backward(buf215, mul_301, view_119, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_301
        del view_119
        buf218 = buf217[0]
        buf227 = reinterpret_tensor(buf186, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf186  # reuse
        buf228 = reinterpret_tensor(buf227, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf227  # reuse
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_12.run(buf228, buf174, buf218, buf224, primals_125, convolution_52, convolution_54, 12288, 144, grid=grid(12288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf230 = aten.convolution_backward(buf228, relu_7, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_214
        buf231 = buf230[0]
        buf233 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf233, relu_7, 6144, grid=grid(6144), stream=stream0)
        del relu_7
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf235 = aten.convolution_backward(buf233, mean_7, primals_212, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_7
        del primals_212
        buf236 = buf235[0]
        buf238 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.div, aten.gelu_backward, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_gelu_backward_mul_sigmoid_13.run(buf174, buf218, buf224, primals_125, convolution_54, buf236, buf238, 1769472, grid=grid(1769472), stream=stream0)
        del primals_125
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf240 = aten.convolution_backward(buf238, mul_289, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_289
        del view_116
        buf241 = buf240[0]
        buf247 = buf241; del buf241  # reuse
        # Source Nodes: [gelu_34], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf247, convolution_51, 884736, grid=grid(884736), stream=stream0)
        del convolution_51
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf249 = aten.convolution_backward(buf247, mul_282, view_113, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_282
        del view_113
        buf250 = buf249[0]
        buf256 = buf250; del buf250  # reuse
        # Source Nodes: [gelu_33], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf256, convolution_50, 884736, grid=grid(884736), stream=stream0)
        del convolution_50
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf258 = aten.convolution_backward(buf256, mul_275, view_110, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_275
        del view_110
        buf259 = buf258[0]
        buf265 = buf259; del buf259  # reuse
        # Source Nodes: [gelu_32], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf265, convolution_49, 884736, grid=grid(884736), stream=stream0)
        del convolution_49
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf267 = aten.convolution_backward(buf265, mul_268, view_107, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_268
        del view_107
        buf268 = buf267[0]
        buf274 = buf2; del buf2  # reuse
        buf277 = reinterpret_tensor(buf236, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf236  # reuse
        buf278 = reinterpret_tensor(buf277, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf277  # reuse
        # Source Nodes: [gelu_31, sigmoid_6], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_14.run(buf274, buf278, buf174, buf218, buf224, buf268, primals_112, convolution_46, convolution_48, 12288, 144, grid=grid(12288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf280 = aten.convolution_backward(buf278, relu_6, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_210
        buf281 = buf280[0]
        buf283 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf283, relu_6, 6144, grid=grid(6144), stream=stream0)
        del relu_6
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf285 = aten.convolution_backward(buf283, mean_6, primals_208, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_6
        del primals_208
        buf286 = buf285[0]
        buf288 = buf268; del buf268  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_10.run(buf274, primals_112, convolution_48, buf286, buf288, 1769472, grid=grid(1769472), stream=stream0)
        del primals_112
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf290 = aten.convolution_backward(buf288, mul_256, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_256
        del view_104
        buf291 = buf290[0]
        buf297 = buf291; del buf291  # reuse
        # Source Nodes: [gelu_30], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf297, convolution_45, 884736, grid=grid(884736), stream=stream0)
        del convolution_45
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf299 = aten.convolution_backward(buf297, mul_249, view_101, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_249
        del view_101
        buf300 = buf299[0]
        buf306 = buf300; del buf300  # reuse
        # Source Nodes: [gelu_29], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf306, convolution_44, 884736, grid=grid(884736), stream=stream0)
        del convolution_44
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf308 = aten.convolution_backward(buf306, mul_242, view_98, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_242
        del view_98
        buf309 = buf308[0]
        buf315 = buf309; del buf309  # reuse
        # Source Nodes: [gelu_28], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf315, convolution_43, 884736, grid=grid(884736), stream=stream0)
        del convolution_43
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf317 = aten.convolution_backward(buf315, mul_235, view_95, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_235
        del view_95
        buf318 = buf317[0]
        buf327 = reinterpret_tensor(buf286, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf286  # reuse
        buf328 = reinterpret_tensor(buf327, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf327  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_15.run(buf328, buf274, buf318, buf324, primals_99, convolution_40, convolution_42, 12288, 144, grid=grid(12288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf330 = aten.convolution_backward(buf328, relu_5, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_206
        buf331 = buf330[0]
        buf333 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf333, relu_5, 6144, grid=grid(6144), stream=stream0)
        del relu_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf335 = aten.convolution_backward(buf333, mean_5, primals_204, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_5
        del primals_204
        buf336 = buf335[0]
        buf338 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.gelu_backward, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_gelu_backward_mul_sigmoid_16.run(buf274, buf318, buf324, primals_99, convolution_42, buf336, buf338, 1769472, grid=grid(1769472), stream=stream0)
        del primals_99
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf340 = aten.convolution_backward(buf338, mul_223, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_223
        del view_92
        buf341 = buf340[0]
        buf347 = buf341; del buf341  # reuse
        # Source Nodes: [gelu_26], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf347, convolution_39, 884736, grid=grid(884736), stream=stream0)
        del convolution_39
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf349 = aten.convolution_backward(buf347, mul_216, view_89, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_216
        del view_89
        buf350 = buf349[0]
        buf356 = buf350; del buf350  # reuse
        # Source Nodes: [gelu_25], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf356, convolution_38, 884736, grid=grid(884736), stream=stream0)
        del convolution_38
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf358 = aten.convolution_backward(buf356, mul_209, view_86, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_209
        del view_86
        buf359 = buf358[0]
        buf365 = buf359; del buf359  # reuse
        # Source Nodes: [gelu_24], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf365, convolution_37, 884736, grid=grid(884736), stream=stream0)
        del convolution_37
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf367 = aten.convolution_backward(buf365, mul_202, view_83, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_202
        del view_83
        buf368 = buf367[0]
        buf374 = buf1; del buf1  # reuse
        buf375 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_23, mul_44, out_37, sigmoid_4], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_17.run(buf374, buf274, buf318, buf324, buf368, convolution_34, convolution_36, buf375, 216, 8192, grid=grid(216), stream=stream0)
        buf377 = reinterpret_tensor(buf336, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf336  # reuse
        buf378 = reinterpret_tensor(buf377, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf377  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_mul_sigmoid_sigmoid_backward_sum_18.run(buf378, buf374, primals_86, convolution_34, convolution_36, 12288, 144, grid=grid(12288), stream=stream0)
        del convolution_34
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf380 = aten.convolution_backward(buf378, relu_4, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_202
        buf381 = buf380[0]
        buf383 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf383, relu_4, 6144, grid=grid(6144), stream=stream0)
        del relu_4
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf385 = aten.convolution_backward(buf383, mean_4, primals_200, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_4
        del primals_200
        buf386 = buf385[0]
        buf388 = buf368; del buf368  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_10.run(buf374, primals_86, convolution_36, buf386, buf388, 1769472, grid=grid(1769472), stream=stream0)
        del convolution_36
        del primals_86
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf390 = aten.convolution_backward(buf388, mul_190, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_190
        del view_80
        buf391 = buf390[0]
        buf397 = buf391; del buf391  # reuse
        # Source Nodes: [gelu_22], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf397, convolution_33, 884736, grid=grid(884736), stream=stream0)
        del convolution_33
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf399 = aten.convolution_backward(buf397, mul_183, view_77, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_183
        del view_77
        buf400 = buf399[0]
        buf406 = buf400; del buf400  # reuse
        # Source Nodes: [gelu_21], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf406, convolution_32, 884736, grid=grid(884736), stream=stream0)
        del convolution_32
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf408 = aten.convolution_backward(buf406, mul_176, view_74, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_176
        del view_74
        buf409 = buf408[0]
        buf415 = buf409; del buf409  # reuse
        # Source Nodes: [gelu_20], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf415, convolution_31, 884736, grid=grid(884736), stream=stream0)
        del convolution_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf417 = aten.convolution_backward(buf415, mul_169, view_71, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_169
        del view_71
        buf418 = buf417[0]
        buf425 = buf374; del buf374  # reuse
        buf428 = reinterpret_tensor(buf386, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf386  # reuse
        buf429 = reinterpret_tensor(buf428, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf428  # reuse
        # Source Nodes: [gelu_19, mul_36, mul_38, mul__22, out_29, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_19.run(buf425, buf429, convolution_28, convolution_30, primals_73, convolution_24, buf418, 12288, 144, grid=grid(12288), stream=stream0)
        del convolution_24
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf431 = aten.convolution_backward(buf429, relu_3, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_198
        buf432 = buf431[0]
        buf434 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_2.run(buf434, relu_3, 6144, grid=grid(6144), stream=stream0)
        del relu_3
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf436 = aten.convolution_backward(buf434, mean_3, primals_196, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_196
        buf437 = buf436[0]
        buf439 = buf418; del buf418  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_10.run(buf425, primals_73, convolution_30, buf437, buf439, 1769472, grid=grid(1769472), stream=stream0)
        del buf437
        del primals_73
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf441 = aten.convolution_backward(buf439, mul_157, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_157
        del view_68
        buf442 = buf441[0]
        buf448 = buf442; del buf442  # reuse
        # Source Nodes: [gelu_18], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf448, convolution_27, 884736, grid=grid(884736), stream=stream0)
        del convolution_27
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf450 = aten.convolution_backward(buf448, mul_150, view_65, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False])
        del mul_150
        del view_65
        buf451 = buf450[0]
        buf457 = buf451; del buf451  # reuse
        # Source Nodes: [gelu_17], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_11.run(buf457, convolution_26, 884736, grid=grid(884736), stream=stream0)
        del convolution_26
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf459 = aten.convolution_backward(buf457, constant_pad_nd_3, view_62, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, False])
        del constant_pad_nd_3
        del view_62
        buf460 = buf459[0]
        buf466 = empty((8, 768, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_16], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_20.run(buf460, convolution_25, buf466, 3538944, grid=grid(3538944), stream=stream0)
        del buf460
        del convolution_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf468 = aten.convolution_backward(buf466, mul_133, view_59, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_133
        del view_59
        buf469 = buf468[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf476 = aten.convolution_backward(buf425, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_1
        del view_56
        buf477 = buf476[0]
        buf0 = empty((8, 512, 24, 24), device='cuda', dtype=torch.float32)
        buf483 = buf0; del buf0  # reuse
        buf486 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf487 = reinterpret_tensor(buf486, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf486  # reuse
        # Source Nodes: [gelu_15, mul_19, mul_21, mul_27, mul_29, mul__12, mul__17, out_13, out_21, shortcut_4, shortcut_5, sigmoid_1, sigmoid_2], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_21.run(buf483, buf487, convolution_21, convolution_23, primals_57, convolution_15, convolution_17, primals_44, convolution_11, buf469, buf477, 4096, 576, grid=grid(4096), stream=stream0)
        del buf477
        buf6 = empty((1000, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_12, out=buf6)
        del clone_12
        buf7 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_22.run(tangents_1, buf7, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf9 = empty((3072, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf8, buf9, 3072, 288, grid=grid(3072), stream=stream0)
        buf12 = buf10[1]
        del buf10
        buf16 = empty((3072, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf15 = empty((3072, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_24.run(buf12, primals_181, unsqueeze_58, squeeze_113, primals_182, buf16, buf15, 3072, 1536, grid=grid(3072), stream=stream0)
        del primals_181
        del primals_182
        del squeeze_113
        del unsqueeze_58
        buf17 = empty((54, ), device='cuda', dtype=torch.float32)
        buf66 = empty((54, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_47, mul_101, mul_93, out_85, out_93, sigmoid_10, sigmoid_11], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_mul_sigmoid_sum_25.run(buf11, convolution_77, convolution_79, buf60, buf4, convolution_71, convolution_73, buf17, buf66, 54, 8192, grid=grid(54), stream=stream0)
        del buf11
        del buf4
        del buf60
        del convolution_71
        del convolution_73
        del convolution_77
        del convolution_79
        buf18 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_101, out_93, sigmoid_11], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_26.run(buf17, buf18, 1, 54, grid=grid(1), stream=stream0)
        del buf17
        buf21 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf20, buf21, 1536, 8, grid=grid(1536), stream=stream0)
        del buf20
        buf24 = buf22[1]
        del buf22
        buf26 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf25, buf26, 768, 8, grid=grid(768), stream=stream0)
        del buf25
        buf29 = buf27[1]
        del buf27
        buf31 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf30, buf31, 1536, 288, grid=grid(1536), stream=stream0)
        del buf30
        buf34 = buf32[1]
        del buf32
        buf38 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf37 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf34, primals_177, unsqueeze_66, squeeze_111, primals_178, buf38, buf37, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_177
        del primals_178
        del squeeze_111
        del unsqueeze_66
        buf40 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf39, buf40, 768, 288, grid=grid(768), stream=stream0)
        del buf39
        buf43 = buf41[1]
        del buf41
        buf47 = reinterpret_tensor(buf8, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf8  # reuse
        buf46 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf43, primals_174, unsqueeze_74, squeeze_109, primals_175, buf47, buf46, 768, 1152, grid=grid(768), stream=stream0)
        del primals_174
        del primals_175
        del squeeze_109
        del unsqueeze_74
        buf49 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf48, buf49, 768, 288, grid=grid(768), stream=stream0)
        del buf48
        buf52 = buf50[1]
        del buf50
        buf56 = buf43; del buf43  # reuse
        buf55 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf52, primals_171, unsqueeze_82, squeeze_107, primals_172, buf56, buf55, 768, 1152, grid=grid(768), stream=stream0)
        del primals_171
        del primals_172
        del squeeze_107
        del unsqueeze_82
        buf58 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf57, buf58, 768, 288, grid=grid(768), stream=stream0)
        del buf57
        buf61 = buf59[1]
        del buf59
        buf65 = reinterpret_tensor(buf34, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf34  # reuse
        buf64 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf61, primals_168, unsqueeze_90, squeeze_105, primals_169, buf65, buf64, 768, 1536, grid=grid(768), stream=stream0)
        del primals_168
        del primals_169
        del squeeze_105
        del unsqueeze_90
        buf67 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_47, mul_93, out_85, sigmoid_10], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_26.run(buf66, buf67, 1, 54, grid=grid(1), stream=stream0)
        buf70 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf69, buf70, 1536, 8, grid=grid(1536), stream=stream0)
        del buf69
        buf73 = buf71[1]
        del buf71
        buf75 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf74, buf75, 768, 8, grid=grid(768), stream=stream0)
        del buf74
        buf78 = buf76[1]
        del buf76
        buf80 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf79, buf80, 1536, 288, grid=grid(1536), stream=stream0)
        del buf79
        buf83 = buf81[1]
        del buf81
        buf87 = reinterpret_tensor(buf61, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf61  # reuse
        buf86 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf83, primals_164, unsqueeze_98, squeeze_103, primals_165, buf87, buf86, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_164
        del primals_165
        del squeeze_103
        del unsqueeze_98
        buf89 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf88, buf89, 768, 288, grid=grid(768), stream=stream0)
        del buf88
        buf92 = buf90[1]
        del buf90
        buf96 = buf52; del buf52  # reuse
        buf95 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf92, primals_161, unsqueeze_106, squeeze_101, primals_162, buf96, buf95, 768, 1152, grid=grid(768), stream=stream0)
        del primals_161
        del primals_162
        del squeeze_101
        del unsqueeze_106
        buf98 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf97, buf98, 768, 288, grid=grid(768), stream=stream0)
        del buf97
        buf101 = buf99[1]
        del buf99
        buf105 = buf92; del buf92  # reuse
        buf104 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf101, primals_158, unsqueeze_114, squeeze_99, primals_159, buf105, buf104, 768, 1152, grid=grid(768), stream=stream0)
        del primals_158
        del primals_159
        del squeeze_99
        del unsqueeze_114
        buf107 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf106, buf107, 768, 288, grid=grid(768), stream=stream0)
        del buf106
        buf110 = buf108[1]
        del buf108
        buf114 = reinterpret_tensor(buf83, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf83  # reuse
        buf113 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf110, primals_155, unsqueeze_122, squeeze_97, primals_156, buf114, buf113, 768, 1536, grid=grid(768), stream=stream0)
        del primals_155
        del primals_156
        del squeeze_97
        del unsqueeze_122
        buf117 = buf66; del buf66  # reuse
        # Source Nodes: [mul_85, out_77, sigmoid_9], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_mul_sigmoid_sum_34.run(buf116, convolution_65, convolution_67, buf117, 54, 8192, grid=grid(54), stream=stream0)
        del convolution_65
        del convolution_67
        buf118 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_85, out_77, sigmoid_9], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_26.run(buf117, buf118, 1, 54, grid=grid(1), stream=stream0)
        del buf117
        buf121 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf120, buf121, 1536, 8, grid=grid(1536), stream=stream0)
        del buf120
        buf124 = buf122[1]
        del buf122
        buf126 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf125, buf126, 768, 8, grid=grid(768), stream=stream0)
        del buf125
        buf129 = buf127[1]
        del buf127
        buf131 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf130, buf131, 1536, 288, grid=grid(1536), stream=stream0)
        del buf130
        buf134 = buf132[1]
        del buf132
        buf138 = reinterpret_tensor(buf110, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf110  # reuse
        buf137 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf134, primals_151, unsqueeze_130, squeeze_95, primals_152, buf138, buf137, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_151
        del primals_152
        del squeeze_95
        del unsqueeze_130
        buf140 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf139, buf140, 768, 288, grid=grid(768), stream=stream0)
        del buf139
        buf143 = buf141[1]
        del buf141
        buf147 = buf101; del buf101  # reuse
        buf146 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf143, primals_148, unsqueeze_138, squeeze_93, primals_149, buf147, buf146, 768, 1152, grid=grid(768), stream=stream0)
        del primals_148
        del primals_149
        del squeeze_93
        del unsqueeze_138
        buf149 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_31.run(buf148, buf149, 768, 288, grid=grid(768), stream=stream0)
        del buf148
        buf152 = buf150[1]
        del buf150
        buf156 = buf143; del buf143  # reuse
        buf155 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf152, primals_145, unsqueeze_146, squeeze_91, primals_146, buf156, buf155, 768, 1152, grid=grid(768), stream=stream0)
        del buf152
        del primals_145
        del primals_146
        del squeeze_91
        del unsqueeze_146
        buf158 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf157, buf158, 768, 1152, grid=grid(768), stream=stream0)
        del buf157
        buf161 = buf159[1]
        del buf159
        buf165 = reinterpret_tensor(buf134, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf134  # reuse
        buf164 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf161, primals_142, unsqueeze_154, squeeze_89, primals_143, buf165, buf164, 768, 1536, grid=grid(768), stream=stream0)
        del primals_142
        del primals_143
        del squeeze_89
        del unsqueeze_154
        buf166 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_29.run(buf116, buf166, 1536, 288, grid=grid(1536), stream=stream0)
        del buf116
        buf169 = buf167[1]
        del buf167
        buf173 = reinterpret_tensor(buf469, (1536, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf469  # reuse
        buf172 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_36.run(buf169, primals_139, unsqueeze_162, squeeze_87, primals_140, buf173, buf172, 1536, 1536, grid=grid(1536), stream=stream0)
        del primals_139
        del primals_140
        del squeeze_87
        del unsqueeze_162
        buf175 = empty((216, ), device='cuda', dtype=torch.float32)
        buf225 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_68, mul_76, out_61, out_69, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_add_gelu_backward_mul_sigmoid_sum_37.run(buf174, convolution_58, convolution_60, buf218, buf224, convolution_52, convolution_54, buf175, buf225, 216, 8192, grid=grid(216), stream=stream0)
        del buf174
        del buf218
        del buf224
        del convolution_52
        del convolution_54
        del convolution_58
        del convolution_60
        buf176 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_76, out_69, sigmoid_8], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_38.run(buf175, buf176, 1, 216, grid=grid(1), stream=stream0)
        buf179 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf178, buf179, 1536, 8, grid=grid(1536), stream=stream0)
        del buf178
        buf182 = buf180[1]
        del buf180
        buf184 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf183, buf184, 768, 8, grid=grid(768), stream=stream0)
        del buf183
        buf187 = buf185[1]
        del buf185
        buf189 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf188, buf189, 1536, 1152, grid=grid(1536), stream=stream0)
        del buf188
        buf192 = buf190[1]
        del buf190
        buf196 = reinterpret_tensor(buf161, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf161  # reuse
        buf195 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf192, primals_135, unsqueeze_170, squeeze_85, primals_136, buf196, buf195, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_135
        del primals_136
        del squeeze_85
        del unsqueeze_170
        buf198 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf197, buf198, 768, 1152, grid=grid(768), stream=stream0)
        buf201 = buf199[1]
        del buf199
        buf205 = reinterpret_tensor(buf197, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf197  # reuse
        buf204 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf201, primals_132, unsqueeze_178, squeeze_83, primals_133, buf205, buf204, 768, 1152, grid=grid(768), stream=stream0)
        del buf201
        del primals_132
        del primals_133
        del squeeze_83
        del unsqueeze_178
        buf207 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf206, buf207, 768, 1152, grid=grid(768), stream=stream0)
        buf210 = buf208[1]
        del buf208
        buf214 = reinterpret_tensor(buf206, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf206  # reuse
        buf213 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf210, primals_129, unsqueeze_186, squeeze_81, primals_130, buf214, buf213, 768, 1152, grid=grid(768), stream=stream0)
        del buf210
        del primals_129
        del primals_130
        del squeeze_81
        del unsqueeze_186
        buf216 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf215, buf216, 768, 1152, grid=grid(768), stream=stream0)
        del buf215
        buf219 = buf217[1]
        del buf217
        buf223 = reinterpret_tensor(buf192, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf192  # reuse
        buf222 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf219, primals_126, unsqueeze_194, squeeze_79, primals_127, buf223, buf222, 768, 1536, grid=grid(768), stream=stream0)
        del primals_126
        del primals_127
        del squeeze_79
        del unsqueeze_194
        buf226 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_68, out_61, sigmoid_7], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_38.run(buf225, buf226, 1, 216, grid=grid(1), stream=stream0)
        buf229 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf228, buf229, 1536, 8, grid=grid(1536), stream=stream0)
        del buf228
        buf232 = buf230[1]
        del buf230
        buf234 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf233, buf234, 768, 8, grid=grid(768), stream=stream0)
        del buf233
        buf237 = buf235[1]
        del buf235
        buf239 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf238, buf239, 1536, 1152, grid=grid(1536), stream=stream0)
        del buf238
        buf242 = buf240[1]
        del buf240
        buf246 = reinterpret_tensor(buf219, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf219  # reuse
        buf245 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf242, primals_122, unsqueeze_202, squeeze_77, primals_123, buf246, buf245, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_122
        del primals_123
        del squeeze_77
        del unsqueeze_202
        buf248 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf247, buf248, 768, 1152, grid=grid(768), stream=stream0)
        buf251 = buf249[1]
        del buf249
        buf255 = reinterpret_tensor(buf247, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf247  # reuse
        buf254 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf251, primals_119, unsqueeze_210, squeeze_75, primals_120, buf255, buf254, 768, 1152, grid=grid(768), stream=stream0)
        del buf251
        del primals_119
        del primals_120
        del squeeze_75
        del unsqueeze_210
        buf257 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf256, buf257, 768, 1152, grid=grid(768), stream=stream0)
        buf260 = buf258[1]
        del buf258
        buf264 = reinterpret_tensor(buf256, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf256  # reuse
        buf263 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf260, primals_116, unsqueeze_218, squeeze_73, primals_117, buf264, buf263, 768, 1152, grid=grid(768), stream=stream0)
        del buf260
        del primals_116
        del primals_117
        del squeeze_73
        del unsqueeze_218
        buf266 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf265, buf266, 768, 1152, grid=grid(768), stream=stream0)
        del buf265
        buf269 = buf267[1]
        del buf267
        buf273 = reinterpret_tensor(buf242, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf242  # reuse
        buf272 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf269, primals_113, unsqueeze_226, squeeze_71, primals_114, buf273, buf272, 768, 1536, grid=grid(768), stream=stream0)
        del primals_113
        del primals_114
        del squeeze_71
        del unsqueeze_226
        buf275 = buf225; del buf225  # reuse
        buf325 = buf175; del buf175  # reuse
        # Source Nodes: [mul_52, mul_60, out_45, out_53, sigmoid_5, sigmoid_6], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_add_gelu_backward_mul_sigmoid_sum_40.run(buf274, convolution_46, convolution_48, buf318, buf324, convolution_40, convolution_42, buf275, buf325, 216, 8192, grid=grid(216), stream=stream0)
        del buf274
        del buf318
        del buf324
        del convolution_40
        del convolution_42
        del convolution_46
        del convolution_48
        buf276 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_60, out_53, sigmoid_6], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_38.run(buf275, buf276, 1, 216, grid=grid(1), stream=stream0)
        del buf275
        buf279 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf278, buf279, 1536, 8, grid=grid(1536), stream=stream0)
        del buf278
        buf282 = buf280[1]
        del buf280
        buf284 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf283, buf284, 768, 8, grid=grid(768), stream=stream0)
        del buf283
        buf287 = buf285[1]
        del buf285
        buf289 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf288, buf289, 1536, 1152, grid=grid(1536), stream=stream0)
        del buf288
        buf292 = buf290[1]
        del buf290
        buf296 = reinterpret_tensor(buf269, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf269  # reuse
        buf295 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf292, primals_109, unsqueeze_234, squeeze_69, primals_110, buf296, buf295, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_109
        del primals_110
        del squeeze_69
        del unsqueeze_234
        buf298 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf297, buf298, 768, 1152, grid=grid(768), stream=stream0)
        buf301 = buf299[1]
        del buf299
        buf305 = reinterpret_tensor(buf297, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf297  # reuse
        buf304 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf301, primals_106, unsqueeze_242, squeeze_67, primals_107, buf305, buf304, 768, 1152, grid=grid(768), stream=stream0)
        del buf301
        del primals_106
        del primals_107
        del squeeze_67
        del unsqueeze_242
        buf307 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf306, buf307, 768, 1152, grid=grid(768), stream=stream0)
        buf310 = buf308[1]
        del buf308
        buf314 = reinterpret_tensor(buf306, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf306  # reuse
        buf313 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf310, primals_103, unsqueeze_250, squeeze_65, primals_104, buf314, buf313, 768, 1152, grid=grid(768), stream=stream0)
        del buf310
        del primals_103
        del primals_104
        del squeeze_65
        del unsqueeze_250
        buf316 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf315, buf316, 768, 1152, grid=grid(768), stream=stream0)
        del buf315
        buf319 = buf317[1]
        del buf317
        buf323 = reinterpret_tensor(buf292, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf292  # reuse
        buf322 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf319, primals_100, unsqueeze_258, squeeze_63, primals_101, buf323, buf322, 768, 1536, grid=grid(768), stream=stream0)
        del primals_100
        del primals_101
        del squeeze_63
        del unsqueeze_258
        buf326 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_52, out_45, sigmoid_5], Original ATen: [aten.add, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_38.run(buf325, buf326, 1, 216, grid=grid(1), stream=stream0)
        del buf325
        buf329 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf328, buf329, 1536, 8, grid=grid(1536), stream=stream0)
        del buf328
        buf332 = buf330[1]
        del buf330
        buf334 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf333, buf334, 768, 8, grid=grid(768), stream=stream0)
        del buf333
        buf337 = buf335[1]
        del buf335
        buf339 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf338, buf339, 1536, 1152, grid=grid(1536), stream=stream0)
        del buf338
        buf342 = buf340[1]
        del buf340
        buf346 = reinterpret_tensor(buf319, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf319  # reuse
        buf345 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf342, primals_96, unsqueeze_266, squeeze_61, primals_97, buf346, buf345, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_96
        del primals_97
        del squeeze_61
        del unsqueeze_266
        buf348 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf347, buf348, 768, 1152, grid=grid(768), stream=stream0)
        buf351 = buf349[1]
        del buf349
        buf355 = reinterpret_tensor(buf347, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf347  # reuse
        buf354 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf351, primals_93, unsqueeze_274, squeeze_59, primals_94, buf355, buf354, 768, 1152, grid=grid(768), stream=stream0)
        del buf351
        del primals_93
        del primals_94
        del squeeze_59
        del unsqueeze_274
        buf357 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf356, buf357, 768, 1152, grid=grid(768), stream=stream0)
        buf360 = buf358[1]
        del buf358
        buf364 = reinterpret_tensor(buf356, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf356  # reuse
        buf363 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf360, primals_90, unsqueeze_282, squeeze_57, primals_91, buf364, buf363, 768, 1152, grid=grid(768), stream=stream0)
        del buf360
        del primals_90
        del primals_91
        del squeeze_57
        del unsqueeze_282
        buf366 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf365, buf366, 768, 1152, grid=grid(768), stream=stream0)
        del buf365
        buf369 = buf367[1]
        del buf367
        buf373 = reinterpret_tensor(buf342, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf342  # reuse
        buf372 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf369, primals_87, unsqueeze_290, squeeze_55, primals_88, buf373, buf372, 768, 1536, grid=grid(768), stream=stream0)
        del primals_87
        del primals_88
        del squeeze_55
        del unsqueeze_290
        buf376 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_44, out_37, sigmoid_4], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_38.run(buf375, buf376, 1, 216, grid=grid(1), stream=stream0)
        buf379 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf378, buf379, 1536, 8, grid=grid(1536), stream=stream0)
        del buf378
        buf382 = buf380[1]
        del buf380
        buf384 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf383, buf384, 768, 8, grid=grid(768), stream=stream0)
        del buf383
        buf387 = buf385[1]
        del buf385
        buf389 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf388, buf389, 1536, 1152, grid=grid(1536), stream=stream0)
        del buf388
        buf392 = buf390[1]
        del buf390
        buf396 = reinterpret_tensor(buf369, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf369  # reuse
        buf395 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf392, primals_83, unsqueeze_298, squeeze_53, primals_84, buf396, buf395, 1536, 768, grid=grid(1536), stream=stream0)
        del primals_83
        del primals_84
        del squeeze_53
        del unsqueeze_298
        buf398 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf397, buf398, 768, 1152, grid=grid(768), stream=stream0)
        buf401 = buf399[1]
        del buf399
        buf405 = reinterpret_tensor(buf397, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf397  # reuse
        buf404 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf401, primals_80, unsqueeze_306, squeeze_51, primals_81, buf405, buf404, 768, 1152, grid=grid(768), stream=stream0)
        del buf401
        del primals_80
        del primals_81
        del squeeze_51
        del unsqueeze_306
        buf407 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf406, buf407, 768, 1152, grid=grid(768), stream=stream0)
        buf410 = buf408[1]
        del buf408
        buf414 = reinterpret_tensor(buf406, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf406  # reuse
        buf413 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf410, primals_77, unsqueeze_314, squeeze_49, primals_78, buf414, buf413, 768, 1152, grid=grid(768), stream=stream0)
        del buf410
        del primals_77
        del primals_78
        del squeeze_49
        del unsqueeze_314
        buf416 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf415, buf416, 768, 1152, grid=grid(768), stream=stream0)
        del buf415
        buf419 = buf417[1]
        del buf417
        buf423 = reinterpret_tensor(buf392, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf392  # reuse
        buf422 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_33.run(buf419, primals_74, unsqueeze_322, squeeze_47, primals_75, buf423, buf422, 768, 1536, grid=grid(768), stream=stream0)
        del primals_74
        del primals_75
        del squeeze_47
        del unsqueeze_322
        buf426 = buf375; del buf375  # reuse
        # Source Nodes: [mul_36, out_29, sigmoid_3], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_mul_sigmoid_sum_41.run(buf425, convolution_28, convolution_30, buf426, 216, 8192, grid=grid(216), stream=stream0)
        del convolution_28
        del convolution_30
        buf427 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_36, out_29, sigmoid_3], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_38.run(buf426, buf427, 1, 216, grid=grid(1), stream=stream0)
        del buf426
        buf430 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_27.run(buf429, buf430, 1536, 8, grid=grid(1536), stream=stream0)
        del buf429
        buf433 = buf431[1]
        del buf431
        buf435 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf434, buf435, 768, 8, grid=grid(768), stream=stream0)
        del buf434
        buf438 = buf436[1]
        del buf436
        buf440 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf439, buf440, 1536, 1152, grid=grid(1536), stream=stream0)
        del buf439
        buf443 = buf441[1]
        del buf441
        buf447 = reinterpret_tensor(buf419, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf419  # reuse
        buf446 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_30.run(buf443, primals_70, unsqueeze_330, squeeze_45, primals_71, buf447, buf446, 1536, 768, grid=grid(1536), stream=stream0)
        del buf443
        del primals_70
        del primals_71
        del squeeze_45
        del unsqueeze_330
        buf449 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf448, buf449, 768, 1152, grid=grid(768), stream=stream0)
        buf452 = buf450[1]
        del buf450
        buf456 = reinterpret_tensor(buf448, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf448  # reuse
        buf455 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf452, primals_67, unsqueeze_338, squeeze_43, primals_68, buf456, buf455, 768, 1152, grid=grid(768), stream=stream0)
        del buf452
        del primals_67
        del primals_68
        del squeeze_43
        del unsqueeze_338
        buf458 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_35.run(buf457, buf458, 768, 1152, grid=grid(768), stream=stream0)
        buf461 = buf459[1]
        del buf459
        buf465 = reinterpret_tensor(buf457, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf457  # reuse
        buf464 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_32.run(buf461, primals_64, unsqueeze_346, squeeze_41, primals_65, buf465, buf464, 768, 1152, grid=grid(768), stream=stream0)
        del buf461
        del primals_64
        del primals_65
        del squeeze_41
        del unsqueeze_346
        buf467 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf466, buf467, 768, 4608, grid=grid(768), stream=stream0)
        del buf466
        buf470 = buf468[1]
        del buf468
        buf474 = empty((768, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf473 = empty((768, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_43.run(buf470, primals_61, unsqueeze_354, squeeze_39, primals_62, buf474, buf473, 768, 512, grid=grid(768), stream=stream0)
        del buf470
        del primals_61
        del primals_62
        del squeeze_39
        del unsqueeze_354
        buf475 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf425, buf475, 1536, 1152, grid=grid(1536), stream=stream0)
        del buf425
        buf478 = buf476[1]
        del buf476
        buf482 = empty((1536, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf481 = empty((1536, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_44.run(buf478, primals_58, unsqueeze_362, squeeze_37, primals_59, buf482, buf481, 1536, 512, grid=grid(1536), stream=stream0)
        del buf478
        del primals_58
        del primals_59
        del squeeze_37
        del unsqueeze_362
        buf484 = empty((288, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_27, out_21, sigmoid_2], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_mul_sigmoid_sum_45.run(buf483, convolution_21, convolution_23, buf484, 288, 8192, grid=grid(288), stream=stream0)
        del convolution_21
        buf485 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_27, out_21, sigmoid_2], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_46.run(buf484, buf485, 1, 288, grid=grid(1), stream=stream0)
        buf488 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_47.run(buf487, buf488, 512, 8, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf489 = aten.convolution_backward(buf487, relu_2, primals_194, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf487
        del primals_194
        buf490 = buf489[0]
        buf491 = buf489[1]
        del buf489
        buf492 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_48.run(buf492, relu_2, 2048, grid=grid(2048), stream=stream0)
        del relu_2
        buf493 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_49.run(buf492, buf493, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf494 = aten.convolution_backward(buf492, mean_2, primals_192, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf492
        del mean_2
        del primals_192
        buf495 = buf494[0]
        buf496 = buf494[1]
        del buf494
        buf497 = reinterpret_tensor(buf169, (8, 512, 24, 24), (294912, 576, 24, 1), 0); del buf169  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_50.run(buf483, primals_57, convolution_23, buf495, buf497, 2359296, grid=grid(2359296), stream=stream0)
        del convolution_23
        del primals_57
        buf498 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_51.run(buf497, buf498, 512, 4608, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf499 = aten.convolution_backward(buf497, mul_121, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf497
        del mul_121
        del view_53
        buf500 = buf499[0]
        buf501 = buf499[1]
        del buf499
        buf505 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf504 = empty((512, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_52.run(buf501, primals_54, unsqueeze_370, squeeze_35, primals_55, buf505, buf504, 512, 256, grid=grid(512), stream=stream0)
        del primals_54
        del primals_55
        del squeeze_35
        del unsqueeze_370
        buf506 = buf500; del buf500  # reuse
        # Source Nodes: [gelu_14], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_53.run(buf506, convolution_20, 1179648, grid=grid(1179648), stream=stream0)
        del convolution_20
        buf507 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf506, buf507, 256, 4608, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf508 = aten.convolution_backward(buf506, mul_114, view_50, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf506
        del mul_114
        del view_50
        buf509 = buf508[0]
        buf510 = buf508[1]
        del buf508
        buf514 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf513 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_55.run(buf510, primals_51, unsqueeze_378, squeeze_33, primals_52, buf514, buf513, 256, 1152, grid=grid(256), stream=stream0)
        del primals_51
        del primals_52
        del squeeze_33
        del unsqueeze_378
        buf515 = buf509; del buf509  # reuse
        # Source Nodes: [gelu_13], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_53.run(buf515, convolution_19, 1179648, grid=grid(1179648), stream=stream0)
        del convolution_19
        buf516 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf515, buf516, 256, 4608, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf517 = aten.convolution_backward(buf515, mul_107, view_47, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf515
        del mul_107
        del view_47
        buf518 = buf517[0]
        buf519 = buf517[1]
        del buf517
        buf523 = buf510; del buf510  # reuse
        buf522 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_55.run(buf519, primals_48, unsqueeze_386, squeeze_31, primals_49, buf523, buf522, 256, 1152, grid=grid(256), stream=stream0)
        del primals_48
        del primals_49
        del squeeze_31
        del unsqueeze_386
        buf524 = buf518; del buf518  # reuse
        # Source Nodes: [gelu_12], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_53.run(buf524, convolution_18, 1179648, grid=grid(1179648), stream=stream0)
        del convolution_18
        buf525 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf524, buf525, 256, 4608, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf526 = aten.convolution_backward(buf524, mul_100, view_44, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf524
        del mul_100
        del view_44
        buf527 = buf526[0]
        buf528 = buf526[1]
        del buf526
        buf532 = reinterpret_tensor(buf501, (256, 512, 1, 1), (512, 1, 1, 1), 0); del buf501  # reuse
        buf531 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_56.run(buf528, primals_45, unsqueeze_394, squeeze_29, primals_46, buf532, buf531, 256, 512, grid=grid(256), stream=stream0)
        del primals_45
        del primals_46
        del squeeze_29
        del unsqueeze_394
        buf534 = buf483; del buf483  # reuse
        buf537 = reinterpret_tensor(buf495, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf495  # reuse
        buf538 = reinterpret_tensor(buf537, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf537  # reuse
        # Source Nodes: [gelu_11, mul_19, mul_21, mul__12, out_13, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_57.run(buf534, buf538, convolution_15, convolution_17, primals_44, convolution_11, buf527, 4096, 576, grid=grid(4096), stream=stream0)
        del convolution_11
        buf535 = buf484; del buf484  # reuse
        # Source Nodes: [mul_19, out_13, sigmoid_1], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_mul_sigmoid_sum_45.run(buf534, convolution_15, convolution_17, buf535, 288, 8192, grid=grid(288), stream=stream0)
        del convolution_15
        buf536 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_19, out_13, sigmoid_1], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_46.run(buf535, buf536, 1, 288, grid=grid(1), stream=stream0)
        buf539 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_47.run(buf538, buf539, 512, 8, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf540 = aten.convolution_backward(buf538, relu_1, primals_190, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf538
        del primals_190
        buf541 = buf540[0]
        buf542 = buf540[1]
        del buf540
        buf543 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_48.run(buf543, relu_1, 2048, grid=grid(2048), stream=stream0)
        del relu_1
        buf544 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_49.run(buf543, buf544, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf545 = aten.convolution_backward(buf543, mean_1, primals_188, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_188
        buf546 = buf545[0]
        buf547 = buf545[1]
        del buf545
        buf548 = buf527; del buf527  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_50.run(buf534, primals_44, convolution_17, buf546, buf548, 2359296, grid=grid(2359296), stream=stream0)
        del buf546
        del convolution_17
        del primals_44
        buf549 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_51.run(buf548, buf549, 512, 4608, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf550 = aten.convolution_backward(buf548, mul_88, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf548
        del mul_88
        del view_41
        buf551 = buf550[0]
        buf552 = buf550[1]
        del buf550
        buf556 = reinterpret_tensor(buf528, (512, 256, 1, 1), (256, 1, 1, 1), 0); del buf528  # reuse
        buf555 = empty((512, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_52.run(buf552, primals_41, unsqueeze_402, squeeze_27, primals_42, buf556, buf555, 512, 256, grid=grid(512), stream=stream0)
        del primals_41
        del primals_42
        del squeeze_27
        del unsqueeze_402
        buf557 = buf551; del buf551  # reuse
        # Source Nodes: [gelu_10], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_53.run(buf557, convolution_14, 1179648, grid=grid(1179648), stream=stream0)
        del convolution_14
        buf558 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf557, buf558, 256, 4608, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf559 = aten.convolution_backward(buf557, mul_81, view_38, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf557
        del mul_81
        del view_38
        buf560 = buf559[0]
        buf561 = buf559[1]
        del buf559
        buf565 = buf519; del buf519  # reuse
        buf564 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_55.run(buf561, primals_38, unsqueeze_410, squeeze_25, primals_39, buf565, buf564, 256, 1152, grid=grid(256), stream=stream0)
        del primals_38
        del primals_39
        del squeeze_25
        del unsqueeze_410
        buf566 = buf560; del buf560  # reuse
        # Source Nodes: [gelu_9], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_53.run(buf566, convolution_13, 1179648, grid=grid(1179648), stream=stream0)
        del convolution_13
        buf567 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf566, buf567, 256, 4608, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf568 = aten.convolution_backward(buf566, constant_pad_nd_2, view_35, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf566
        del constant_pad_nd_2
        del view_35
        buf569 = buf568[0]
        buf570 = buf568[1]
        del buf568
        buf574 = buf561; del buf561  # reuse
        buf573 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_55.run(buf570, primals_35, unsqueeze_418, squeeze_23, primals_36, buf574, buf573, 256, 1152, grid=grid(256), stream=stream0)
        del buf570
        del primals_35
        del primals_36
        del squeeze_23
        del unsqueeze_418
        buf575 = reinterpret_tensor(buf12, (8, 256, 48, 48), (589824, 2304, 48, 1), 0); del buf12  # reuse
        # Source Nodes: [gelu_8], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_58.run(buf569, convolution_12, buf575, 4718592, grid=grid(4718592), stream=stream0)
        del buf569
        del convolution_12
        buf576 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_59.run(buf575, buf576, 256, 18432, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf577 = aten.convolution_backward(buf575, mul_64, view_32, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_64
        del view_32
        buf578 = buf577[0]
        buf579 = buf577[1]
        del buf577
        buf583 = empty((256, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf582 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_60.run(buf579, primals_32, unsqueeze_426, squeeze_21, primals_33, buf583, buf582, 256, 256, grid=grid(256), stream=stream0)
        del buf579
        del primals_32
        del primals_33
        del squeeze_21
        del unsqueeze_426
        buf584 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_51.run(buf534, buf584, 512, 4608, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf585 = aten.convolution_backward(buf534, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d
        del buf534
        del view_29
        buf586 = buf585[0]
        buf587 = buf585[1]
        del buf585
        buf591 = buf552; del buf552  # reuse
        buf590 = empty((512, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_52.run(buf587, primals_29, unsqueeze_434, squeeze_19, primals_30, buf591, buf590, 512, 256, grid=grid(512), stream=stream0)
        del buf587
        del primals_29
        del primals_30
        del squeeze_19
        del unsqueeze_434
        buf593 = buf578; del buf578  # reuse
        buf596 = reinterpret_tensor(buf543, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf543  # reuse
        buf597 = reinterpret_tensor(buf596, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf596  # reuse
        # Source Nodes: [gelu_7, mul_10, mul_12, mul__7, out_5, shortcut_2, sigmoid], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.gelu, aten.gelu_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_red_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_61.run(buf593, buf597, convolution_8, convolution_10, primals_28, convolution_4, buf586, 2048, 2304, grid=grid(2048), stream=stream0)
        del buf586
        del convolution_4
        buf594 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_10, out_5, sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_mul_sigmoid_sum_62.run(buf593, convolution_8, convolution_10, buf594, 576, 8192, grid=grid(576), stream=stream0)
        del convolution_8
        buf595 = empty((), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_10, out_5, sigmoid], Original ATen: [aten.mul, aten.sigmoid, aten.sum]
        triton_per_fused_mul_sigmoid_sum_63.run(buf594, buf595, 1, 576, grid=grid(1), stream=stream0)
        buf598 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_49.run(buf597, buf598, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf599 = aten.convolution_backward(buf597, relu, primals_186, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf597
        del primals_186
        buf600 = buf599[0]
        buf601 = buf599[1]
        del buf599
        buf602 = buf600; del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_64.run(buf602, relu, 1024, grid=grid(1024), stream=stream0)
        del relu
        buf603 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_65.run(buf602, buf603, 128, 8, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf604 = aten.convolution_backward(buf602, mean, primals_184, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf602
        del mean
        del primals_184
        buf605 = buf604[0]
        buf606 = buf604[1]
        del buf604
        buf607 = buf575; del buf575  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.sigmoid]
        triton_poi_fused_add_div_mul_sigmoid_66.run(buf593, primals_28, convolution_10, buf605, buf607, 4718592, grid=grid(4718592), stream=stream0)
        del buf605
        del convolution_10
        del primals_28
        buf608 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_59.run(buf607, buf608, 256, 18432, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf609 = aten.convolution_backward(buf607, mul_52, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf607
        del mul_52
        del view_26
        buf610 = buf609[0]
        buf611 = buf609[1]
        del buf609
        buf615 = empty((256, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf614 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_67.run(buf611, primals_25, unsqueeze_442, squeeze_17, primals_26, buf615, buf614, 256, 128, grid=grid(256), stream=stream0)
        del primals_25
        del primals_26
        del squeeze_17
        del unsqueeze_442
        buf616 = buf610; del buf610  # reuse
        # Source Nodes: [gelu_6], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_68.run(buf616, convolution_7, 2359296, grid=grid(2359296), stream=stream0)
        del convolution_7
        buf617 = empty_strided((128, 3), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_69.run(buf616, buf617, 384, 6144, grid=grid(384), stream=stream0)
        buf618 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_70.run(buf617, buf618, 128, 3, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf619 = aten.convolution_backward(buf616, mul_45, view_23, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf616
        del mul_45
        del view_23
        buf620 = buf619[0]
        buf621 = buf619[1]
        del buf619
        buf625 = empty((128, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf624 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_71.run(buf621, primals_22, unsqueeze_450, squeeze_15, primals_23, buf625, buf624, 128, 1152, grid=grid(128), stream=stream0)
        del primals_22
        del primals_23
        del squeeze_15
        del unsqueeze_450
        buf626 = buf620; del buf620  # reuse
        # Source Nodes: [gelu_5], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_68.run(buf626, convolution_6, 2359296, grid=grid(2359296), stream=stream0)
        del convolution_6
        buf627 = buf617; del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_69.run(buf626, buf627, 384, 6144, grid=grid(384), stream=stream0)
        buf628 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_70.run(buf627, buf628, 128, 3, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf629 = aten.convolution_backward(buf626, mul_38, view_20, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf626
        del mul_38
        del view_20
        buf630 = buf629[0]
        buf631 = buf629[1]
        del buf629
        buf635 = buf621; del buf621  # reuse
        buf634 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_red_fused_mul_native_batch_norm_backward_view_71.run(buf631, primals_19, unsqueeze_458, squeeze_13, primals_20, buf635, buf634, 128, 1152, grid=grid(128), stream=stream0)
        del buf631
        del primals_19
        del primals_20
        del squeeze_13
        del unsqueeze_458
        buf636 = buf630; del buf630  # reuse
        # Source Nodes: [gelu_4], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_68.run(buf636, convolution_5, 2359296, grid=grid(2359296), stream=stream0)
        del convolution_5
        buf637 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_69.run(buf636, buf637, 384, 6144, grid=grid(384), stream=stream0)
        buf638 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_70.run(buf637, buf638, 128, 3, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf639 = aten.convolution_backward(buf636, mul_28, view_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf636
        del view_17
        buf640 = buf639[0]
        buf641 = buf639[1]
        del buf639
        buf645 = empty((128, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf644 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_72.run(buf641, primals_16, unsqueeze_466, squeeze_11, primals_17, buf645, buf644, 128, 128, grid=grid(128), stream=stream0)
        del buf641
        del primals_16
        del primals_17
        del squeeze_11
        del unsqueeze_466
        buf646 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_59.run(buf593, buf646, 256, 18432, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf647 = aten.convolution_backward(buf593, mul_28, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_28
        del view_14
        buf648 = buf647[0]
        buf649 = buf647[1]
        del buf647
        buf653 = buf611; del buf611  # reuse
        buf652 = empty((256, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_67.run(buf649, primals_13, unsqueeze_474, squeeze_9, primals_14, buf653, buf652, 256, 128, grid=grid(256), stream=stream0)
        del buf649
        del primals_13
        del primals_14
        del squeeze_9
        del unsqueeze_474
        buf654 = buf640; del buf640  # reuse
        # Source Nodes: [gelu_3], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_add_gelu_gelu_backward_mul_73.run(buf654, buf648, convolution_3, 2359296, grid=grid(2359296), stream=stream0)
        del buf648
        del convolution_3
        buf655 = buf637; del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_69.run(buf654, buf655, 384, 6144, grid=grid(384), stream=stream0)
        buf656 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_70.run(buf655, buf656, 128, 3, grid=grid(128), stream=stream0)
        del buf655
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf657 = aten.convolution_backward(buf654, constant_pad_nd_1, view_11, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf654
        del constant_pad_nd_1
        del view_11
        buf658 = buf657[0]
        buf659 = buf657[1]
        del buf657
        buf663 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf662 = empty((128, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_74.run(buf659, primals_10, unsqueeze_482, squeeze_7, primals_11, buf663, buf662, 128, 576, grid=grid(128), stream=stream0)
        del buf659
        del primals_10
        del primals_11
        del squeeze_7
        del unsqueeze_482
        buf664 = reinterpret_tensor(buf593, (8, 64, 96, 96), (589824, 9216, 96, 1), 0); del buf593  # reuse
        # Source Nodes: [gelu_2], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_75.run(buf658, convolution_2, buf664, 4718592, grid=grid(4718592), stream=stream0)
        del buf658
        del convolution_2
        buf665 = reinterpret_tensor(buf594, (64, 9), (9, 1), 0); del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_76.run(buf664, buf665, 576, 8192, grid=grid(576), stream=stream0)
        buf666 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_77.run(buf665, buf666, 64, 9, grid=grid(64), stream=stream0)
        del buf665
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf667 = aten.convolution_backward(buf664, mul_13, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf664
        del mul_13
        del view_8
        buf668 = buf667[0]
        buf669 = buf667[1]
        del buf667
        buf673 = empty((64, 32, 3, 3), device='cuda', dtype=torch.float32)
        buf672 = empty((64, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_78.run(buf669, primals_7, unsqueeze_490, squeeze_5, primals_8, buf673, buf672, 64, 288, grid=grid(64), stream=stream0)
        del buf669
        del primals_7
        del primals_8
        del squeeze_5
        del unsqueeze_490
        buf674 = buf668; del buf668  # reuse
        # Source Nodes: [gelu_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_68.run(buf674, convolution_1, 2359296, grid=grid(2359296), stream=stream0)
        del convolution_1
        buf675 = reinterpret_tensor(buf535, (32, 9), (9, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_79.run(buf674, buf675, 288, 8192, grid=grid(288), stream=stream0)
        buf676 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_80.run(buf675, buf676, 32, 9, grid=grid(32), stream=stream0)
        del buf675
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf677 = aten.convolution_backward(buf674, mul_6, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf674
        del mul_6
        del view_5
        buf678 = buf677[0]
        buf679 = buf677[1]
        del buf677
        buf683 = empty((32, 16, 3, 3), device='cuda', dtype=torch.float32)
        buf682 = empty((32, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_81.run(buf679, primals_4, unsqueeze_498, squeeze_3, primals_5, buf683, buf682, 32, 144, grid=grid(32), stream=stream0)
        del buf679
        del primals_4
        del primals_5
        del squeeze_3
        del unsqueeze_498
        buf684 = buf678; del buf678  # reuse
        # Source Nodes: [gelu], Original ATen: [aten.gelu, aten.gelu_backward, aten.mul]
        triton_poi_fused_gelu_gelu_backward_mul_53.run(buf684, convolution, 1179648, grid=grid(1179648), stream=stream0)
        del convolution
        buf685 = empty((16, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_82.run(buf684, buf685, 144, 8192, grid=grid(144), stream=stream0)
        buf686 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_83.run(buf685, buf686, 16, 9, grid=grid(16), stream=stream0)
        del buf685
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf687 = aten.convolution_backward(buf684, constant_pad_nd, view_2, [16], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf684
        del constant_pad_nd
        del view_2
        buf688 = buf687[1]
        del buf687
        buf692 = empty((16, 3, 3, 3), device='cuda', dtype=torch.float32)
        buf691 = empty((16, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward, aten.view]
        triton_per_fused_mul_native_batch_norm_backward_view_84.run(buf688, primals_1, unsqueeze_506, squeeze_1, primals_2, buf692, buf691, 16, 27, grid=grid(16), stream=stream0)
        del buf688
        del primals_1
        del primals_2
        del squeeze_1
        del unsqueeze_506
        return (buf692, buf691, buf686, buf683, buf682, buf676, buf673, buf672, buf666, buf663, buf662, buf656, buf653, buf652, buf646, buf645, buf644, buf638, buf635, buf634, buf628, buf625, buf624, buf618, buf615, buf614, buf608, buf595, buf591, buf590, buf584, buf583, buf582, buf576, buf574, buf573, buf567, buf565, buf564, buf558, buf556, buf555, buf549, buf536, buf532, buf531, buf525, buf523, buf522, buf516, buf514, buf513, buf507, buf505, buf504, buf498, buf485, buf482, buf481, buf475, buf474, buf473, buf467, buf465, buf464, buf458, buf456, buf455, buf449, buf447, buf446, buf440, buf427, buf423, buf422, buf416, buf414, buf413, buf407, buf405, buf404, buf398, buf396, buf395, buf389, buf376, buf373, buf372, buf366, buf364, buf363, buf357, buf355, buf354, buf348, buf346, buf345, buf339, buf326, buf323, buf322, buf316, buf314, buf313, buf307, buf305, buf304, buf298, buf296, buf295, buf289, buf276, buf273, buf272, buf266, buf264, buf263, buf257, buf255, buf254, buf248, buf246, buf245, buf239, buf226, buf223, buf222, buf216, buf214, buf213, buf207, buf205, buf204, buf198, buf196, buf195, buf189, buf176, buf173, buf172, buf166, buf165, buf164, buf158, buf156, buf155, buf149, buf147, buf146, buf140, buf138, buf137, buf131, buf118, buf114, buf113, buf107, buf105, buf104, buf98, buf96, buf95, buf89, buf87, buf86, buf80, buf67, buf65, buf64, buf58, buf56, buf55, buf49, buf47, buf46, buf40, buf38, buf37, buf31, buf18, buf16, buf15, buf9, buf606, buf603, buf601, buf598, buf547, buf544, buf542, buf539, buf496, buf493, buf491, buf488, buf438, buf435, buf433, buf430, buf387, buf384, buf382, buf379, buf337, buf334, buf332, buf329, buf287, buf284, buf282, buf279, buf237, buf234, buf232, buf229, buf187, buf184, buf182, buf179, buf129, buf126, buf124, buf121, buf78, buf75, buf73, buf70, buf29, buf26, buf24, buf21, reinterpret_tensor(buf6, (1000, 3072), (3072, 1), 0), reinterpret_tensor(buf7, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((3072, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd = rand_strided((8, 3, 193, 193), (111747, 37249, 193, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 16, 96, 96), (147456, 9216, 96, 1), device='cuda:0', dtype=torch.float32)
    mul_6 = rand_strided((8, 16, 96, 96), (147456, 9216, 96, 1), device='cuda:0', dtype=torch.float32)
    squeeze_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 96, 96), (294912, 9216, 96, 1), device='cuda:0', dtype=torch.float32)
    mul_13 = rand_strided((8, 32, 96, 96), (294912, 9216, 96, 1), device='cuda:0', dtype=torch.float32)
    squeeze_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 96, 96), (589824, 9216, 96, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_1 = rand_strided((8, 64, 97, 97), (602176, 9409, 97, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    mul_28 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    squeeze_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 256, 48, 48), (589824, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    squeeze_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    mul_38 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    squeeze_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    mul_52 = rand_strided((8, 128, 48, 48), (294912, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    squeeze_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 48, 48), (589824, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_64 = rand_strided((8, 256, 48, 48), (589824, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 512, 24, 24), (294912, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 256, 48, 48), (589824, 2304, 48, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_2 = rand_strided((8, 256, 49, 49), (614656, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    squeeze_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    mul_81 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 512, 24, 24), (294912, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_100 = rand_strided((8, 512, 24, 24), (294912, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    mul_114 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    mul_121 = rand_strided((8, 256, 24, 24), (147456, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    squeeze_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 24, 24), (294912, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((8, 512, 24, 24), (294912, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((8, 512, 12, 12), (73728, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_56 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((768, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 768, 24, 24), (442368, 576, 24, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_3 = rand_strided((8, 768, 25, 25), (480000, 625, 25, 1), device='cuda:0', dtype=torch.float32)
    squeeze_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_150 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_45 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_169 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_176 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_183 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_190 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_53 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_80 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_202 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_209 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_216 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_223 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_235 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_242 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_98 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_249 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_256 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_69 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_268 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_275 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_282 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_113 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_289 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_77 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_301 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_308 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_315 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mul_322 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_334 = rand_strided((8, 1536, 12, 12), (221184, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_87 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 768, 12, 12), (110592, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_4 = rand_strided((8, 768, 13, 13), (129792, 169, 13, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_351 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_358 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_95 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_370 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_377 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_384 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_391 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_403 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_158 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_410 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_417 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mul_424 = rand_strided((8, 768, 6, 6), (27648, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_111 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_118 = rand_strided((8, 1536, 6, 6), (55296, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    squeeze_113 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 3072, 6, 6), (110592, 36, 6, 1), device='cuda:0', dtype=torch.float32)
    clone_12 = rand_strided((8, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_58 = rand_strided((1, 3072, 1), (3072, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_66 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_74 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_82 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_90 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_98 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_106 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_114 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_122 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_146 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_170 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_194 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_218 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 768, 1), (768, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_83, primals_84, primals_86, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_156, primals_158, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, constant_pad_nd, squeeze_1, view_2, convolution, mul_6, squeeze_3, view_5, convolution_1, mul_13, squeeze_5, view_8, convolution_2, constant_pad_nd_1, squeeze_7, view_11, convolution_3, mul_28, squeeze_9, view_14, convolution_4, squeeze_11, view_17, convolution_5, mul_38, squeeze_13, view_20, convolution_6, mul_45, squeeze_15, view_23, convolution_7, mul_52, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_64, avg_pool2d, squeeze_19, view_29, convolution_11, squeeze_21, view_32, convolution_12, constant_pad_nd_2, squeeze_23, view_35, convolution_13, mul_81, squeeze_25, view_38, convolution_14, mul_88, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_100, squeeze_29, view_44, convolution_18, mul_107, squeeze_31, view_47, convolution_19, mul_114, squeeze_33, view_50, convolution_20, mul_121, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_133, avg_pool2d_1, squeeze_37, view_56, convolution_24, squeeze_39, view_59, convolution_25, constant_pad_nd_3, squeeze_41, view_62, convolution_26, mul_150, squeeze_43, view_65, convolution_27, mul_157, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_169, squeeze_47, view_71, convolution_31, mul_176, squeeze_49, view_74, convolution_32, mul_183, squeeze_51, view_77, convolution_33, mul_190, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_202, squeeze_55, view_83, convolution_37, mul_209, squeeze_57, view_86, convolution_38, mul_216, squeeze_59, view_89, convolution_39, mul_223, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_235, squeeze_63, view_95, convolution_43, mul_242, squeeze_65, view_98, convolution_44, mul_249, squeeze_67, view_101, convolution_45, mul_256, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_268, squeeze_71, view_107, convolution_49, mul_275, squeeze_73, view_110, convolution_50, mul_282, squeeze_75, view_113, convolution_51, mul_289, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_301, squeeze_79, view_119, convolution_55, mul_308, squeeze_81, view_122, convolution_56, mul_315, squeeze_83, view_125, convolution_57, mul_322, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_334, avg_pool2d_2, squeeze_87, view_131, convolution_61, squeeze_89, view_134, convolution_62, constant_pad_nd_4, squeeze_91, view_137, convolution_63, mul_351, squeeze_93, view_140, convolution_64, mul_358, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_370, squeeze_97, view_146, convolution_68, mul_377, squeeze_99, view_149, convolution_69, mul_384, squeeze_101, view_152, convolution_70, mul_391, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_403, squeeze_105, view_158, convolution_74, mul_410, squeeze_107, view_161, convolution_75, mul_417, squeeze_109, view_164, convolution_76, mul_424, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_118, squeeze_113, view_170, convolution_80, clone_12, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dm_nfnet_f0', benchmark_compiled_module)
