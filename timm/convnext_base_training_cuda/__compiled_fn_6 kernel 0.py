
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtcbt4g57hrviadj7jcv3toxxh4xcnjikjw4nylbesjf4s5onol.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': []}
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpa6h73asf24eby24mqwnsvfcqa36e2axyfonk2plxvzzlpiquk.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csy5h2s6tkrooivuzgf46vhri4upyr5rzowleoajfzq2xumlwtsw.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.native_layer_norm_backward, aten.squeeze]

triton_per_fused_as_strided_scatter_native_layer_norm_backward_squeeze_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_as_strided_scatter_native_layer_norm_backward_squeeze_2', 'mutated_arg_names': ['out_ptr3']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel):
    xnumel = 8
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57d2joyellxev5sfebupqoqmb4sneuy4kjuwq5prh3vlb3ppksd.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eo/ceortceebfoz4usip2qc242rdl3m3i6dzivy5opdojidyo65bbsp.py
# Source Nodes: [], Original ATen: [aten.clone]

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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x2 = (xindex // 50176)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t6/ct6lczhnjv4svdzoz4f4s2stxrdvrhsw4ccjdlziwywam2a64ptg.py
# Source Nodes: [x_515], Original ATen: [aten.gelu, aten.gelu_backward]
# x_515 => add_150, erf_35, mul_221
triton_poi_fused_gelu_gelu_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq75turgdjoyrp4tzxyqkxooz5vxixe7vmddiushazam5lzfxtr5.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2eazlyjkq2ij3yunivbbbwdjtyx74md7wfnzv3xceagyz4vgvnz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    x0 = xindex % 8
    x2 = (xindex // 56) % 7
    x3 = (xindex // 392)
    x7 = xindex % 56
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r4 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x2 + (7*r4) + (896*x7) + (50176*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkouetees46qvvr3maq4c3ogsscw4mefs3arsdlbaundzgnbif2.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (8*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkimiajprabbeslxcysdlmuibepp345nbjxl2yaqpwpkaivmu7q.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 56
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 1024)
    y0 = yindex % 7
    y1 = (yindex // 7)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (y0 + (7*x3) + (49*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3 + (7*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (7*x5) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x3 + (7*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 1024.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr1 + (x2 + (1024*y0) + (7168*x3) + (50176*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2g6ygmusqv4xgw5kmkjvhzyolvsl6fi7e4cekc5muwpbyhk4lme.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]

triton_red_fused_add_div_mul_sum_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((49*x0) + (50176*(r2 // 49)) + (100352*x1) + (r2 % 49)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp9 = tmp2 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirxuax5vymnxl7ox5ky73xrdiqt7odkdl6rcwdxa7gryavai7il.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.sum]

triton_per_fused_div_mul_sum_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_sum_11', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ix/cix4jipjbsxsbfrqplijs3b7je2mnwt2gl7e25ni7xkf3obbvoi6.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqjekeu2o6g3edvfcqheyfnv3hq5edaoivmtscqfnhh5bnsjqx3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (401408*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlkesxjva44oleee4b6uuxajncdft42g6cetjyeta2mcmjbr643.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yt/cytrgnamsrbzlb5rdksj64rt72rukvsaqobz54l7e6tjszaomv6h.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((7*x0) + (7168*(r2 % 7)) + (50176*(r2 // 49)) + (100352*x1) + ((r2 // 7) % 7)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogsgwje5if6y2oehcemyp2meytbhdyvinfvvrvgzpyreu2oef43.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*((r2 // 7) % 7)) + (7168*(r2 % 7)) + (50176*(r2 // 49)) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/cary5tcxkyic2mn6g6aminy6cuukciugq2utenzytz2q7qllthqj.py
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
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (49*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crby3sx52yyn5px4z5tb5olspv2fae7i7jb75sslfh6i4a5gc4om.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 56
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 1024)
    y0 = yindex % 7
    y1 = (yindex // 7)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (y0 + (7*x3) + (49*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3 + (7*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (7*x5) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3 + (7*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 1024.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x2 + (1024*y0) + (7168*x3) + (50176*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxjtlqul3ndkgwxt2kldn5vxvyzdqqmb2zdm25hbba5mubrkuad.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*((r2 // 7) % 7)) + (7168*(r2 % 7)) + (50176*(r2 // 49)) + (100352*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kz/ckz7xj2axj3fxezejdudmk3ahyadsxdebwksoiodwtwecmyfizc4.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]

triton_per_fused_add_div_mul_sum_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = (rindex // 49)
    x0 = xindex
    r1 = rindex % 49
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0 + (1024*r3)), rmask & xmask, other=0.0)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/cteou4pwp22kctusxxoddjkoaenxavvsfsgfvvaov63v3hl5eqhx.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (y3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (y0 + (1024*x2) + (50176*y1)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyylr666xp4axg4byd2x2ax5iyxd5nzww77pmva7invog6jszuh.py
# Source Nodes: [], Original ATen: [aten.add, aten.div]

triton_poi_fused_add_div_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp7 = tl.load(in_ptr2 + (x2), None)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5lhg7ojvapqqndydwpgrqk5nd4ixugnyw2hb5w6ntv4vkrd34z.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65xsftnxh6owa3y4ufqub3nuc2hhwvzwmv4ittxc5im4w4ytskq.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 4
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jx/cjx2v2vlnlxgwjid4ktifo2olt3wsiegcy3gilx2oixytmsoiqdg.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (784*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sd/csdm3kaosgu56msmnfjtmdwvasv63wkw4bzgghdacd7estu7bbwk.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x3 = (xindex // 784)
    x5 = (xindex // 4) % 196
    x2 = (xindex // 56) % 14
    x7 = xindex % 56
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x8 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (x5 + (196*r4) + (25088*x0) + (100352*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x2 + (14*r4) + (1792*x7) + (100352*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x8), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hxjlxen7bz5ungq45tonaw2262lsfosyl3edchouranjynrw6b.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_youkaichao/6v/c6v3lzf3xrjfla535lic4n5qgumk3u3scdciig77rnyjaxqd7t46.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 196)
        r4 = rindex % 196
        r1 = rindex % 14
        r2 = (rindex // 14) % 14
        tmp0 = tl.load(in_ptr0 + (r4 + (196*x0) + (100352*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (14*x0) + (7168*r1) + (100352*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2e4igvr2yvtx2nvlnbx6tdsycdl4sarmp5kxipulxmg3jhzsyz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_poi_fused_native_layer_norm_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 14
    y1 = (yindex // 14) % 14
    y2 = (yindex // 196)
    x3 = xindex
    y4 = yindex % 196
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y1 + (14*y0) + (196*y2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4 + (196*x3) + (100352*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y5), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y1 + (14*x3) + (7168*y0) + (100352*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y5), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 512.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x3 + (512*y5)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/ceb53d5jyq7scikgupxlwtrig5bw3xldhi2ixigmuyyhgeceancb.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnz2f54w6htg7i2bxvw62skolgwcxzqgu367zm2qyeirayt5x5r.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((196*x1) + (100352*(y0 // 196)) + (y0 % 196)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rfimah5kjz57rqa2nzkowz6olphtszxypqvgic42wfrlc6uszt.py
# Source Nodes: [x_468], Original ATen: [aten.gelu, aten.gelu_backward]
# x_468 => add_136, erf_32, mul_201
triton_poi_fused_gelu_gelu_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hw/chwyiji7uf6ip7svbfcr2khmzx4wkmhylrm5cnrmhyfzqerpcumi.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 1568
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
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6o5ha2cfyhf4dupkb7d3cfnmqjmcpc5j6yrnepkuib6moojgb3o.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    x0 = xindex % 4
    x2 = (xindex // 56) % 14
    x3 = (xindex // 784)
    x7 = xindex % 56
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r4 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x2 + (14*r4) + (1792*x7) + (100352*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/cclamk2plf4btk6t4cgizzoyptyprcspz6pbjxqcecapydaja535.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 512)
    y0 = yindex % 14
    y1 = (yindex // 14)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (y0 + (14*x3) + (196*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3 + (14*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (14*x5) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x3 + (14*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 512.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr1 + (x2 + (512*y0) + (7168*x3) + (100352*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbhk4nll2rxpo6bp3kqz4htvshuq7wgengjaqmmlpaswtcmpxap.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr2 + ((196*x0) + (100352*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp3 + tmp11
        tmp13 = tl.load(in_ptr3 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp12 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xi/cxiowlwzxwm5cuh7byia3yj5i7mcpleu7snq2dtvgonicqujkrom.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4prumz45ul7v7kbehij257clpkxudx56nvno277ssmdvryocgf.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*r2) + (61952*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4spftzapnr3yr4pujspcj47tphlrcc733glawe2ddttbubaare.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26624
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*r2) + (247808*x1)), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lt6zgqg7hxn7mcytydf3enndm7cf2tezsvs7msfq3tslr7hxra.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirgzlkvydah4xikgqujamqvhonnskr7kmabti5dl6imf7irutp3.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((14*x0) + (7168*((r2 + (121*x1)) % 14)) + (100352*(((r2 + (121*x1)) // 196) % 8)) + (((r2 + (121*x1)) // 14) % 14)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp12 = tl.where(tmp2, tmp3, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskbt6ppbnrulhsodmc33x7n5aqn23aq5rol2goiqfnhonyl3kcs.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(((r2 + (121*x1)) // 14) % 14)) + (7168*((r2 + (121*x1)) % 14)) + (100352*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mr/cmrrug2xdundj666sycflerq3x6mc5yb3mjydf7n5bun44evijze.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vcszfbffu5i67itk6lxxrrtd4drtr4dmjkgdd7zrxybnwc5gsr.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 512)
    y0 = yindex % 14
    y1 = (yindex // 14)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (y0 + (14*x3) + (196*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3 + (14*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (14*x5) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3 + (14*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 512.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x2 + (512*y0) + (7168*x3) + (100352*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qs/cqshmepufdbj3dfs2ysslkovemlloe2ul75qbunftev36x7otun4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(((r2 + (121*x1)) // 14) % 14)) + (7168*((r2 + (121*x1)) % 14)) + (100352*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwl6pshz7n5qqomy77kngkn4ji5fy3343ys4zxg4rfikgknlrxb.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr3 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnhuzbwtk3pcx2bay2o5adkd4mv3ht3qaqzt72lv7byntbs6nch.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_per_fused_add_mul_sum_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvh7qx46mgcclzqflzo4rltdxaihgmq7cdbditbonqsj2b7g5gwv.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cflvyiuo3smefs575ip7vo3isekrashulgmols45ia24jajy6eex.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/cekeeleepnjjgklpw576bsj5ghjojz2mszqrk56jsc2sjfdebqxp.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzdejuzgyftijvvkhk26fmhss4g2wusgbada6vq2xw7apx47bad.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6mtpgdghdkaz7l7l2snbuu252jxhc54zwel3kwbrjyvblhyrvw.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (100352*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (512*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkgbar25ln7knkexic47ntd252hx3mtcqwfddxdh6cyilyw4l42.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cds6c54h3znntel2mnlyjiazmu2oy2pmb4yistpfewdhqst66rzg.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jnjzxnyjz2qk2ifmgn3cm26w73zlxonoj6vgie3q6o56odbuae.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr5 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr6 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp2 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp16 = tmp9 + tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/53/c53pjbcgv5giqb4iqb3igu5ye2syfolrzejjlqt5ohkke4yj4gim.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn24pwywl2dohogygpaxusn2ietgys7j4ari3smh5wji73zhvbtc.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 512
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskgacvjr2ipiwjz44ut3byllej56thsqrhsz5fplfjabqn4cjdf.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp2 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsauncnd3j5n6ifkogqzvhj4dy3i36ncv6uilimgffymrocgkso.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x2 = xindex % 14
    x3 = (xindex // 14)
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x5 + (196*y4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + (196*y4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 + (196*y4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x5 + (196*y4)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (y0 + (512*x3) + (7168*x2) + (100352*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxfie22yraoxujokhby5kqqzoeuntmq6gn556qwmbqu5cdywmaj.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 2
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r3) + (100352*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5q3ohwkxp2vpkscals2mjavikwbptlqpk4vbi5oupdvuxhffn7d.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs634uwukugmwlv7n4djmqzshkpf2t5lxw5vakhm45jvo3k6j5bf.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x3 = (xindex // 1568)
    x5 = (xindex // 2) % 784
    x2 = (xindex // 56) % 28
    x7 = xindex % 56
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x8 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (x5 + (784*r4) + (100352*x0) + (200704*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x2 + (28*r4) + (3584*x7) + (200704*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x8), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27ac6aayydms7zk2kz2nejtjg4b4p2zffxuxccamuamoq7ssdmi.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbderfvq7wpnaayyzpk77fjujj7632lbfshr4f75rgxzvdhzwpo5.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 784)
        r4 = rindex % 784
        r1 = rindex % 28
        r2 = (rindex // 28) % 28
        tmp0 = tl.load(in_ptr0 + (r4 + (784*x0) + (200704*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (28*x0) + (7168*r1) + (200704*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kpzplus2si7whj5a6d3wkrgvmdgkdfrlhujezavx52vucprzo3.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_poi_fused_native_layer_norm_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 256)
    y0 = yindex % 28
    y1 = (yindex // 28)
    x2 = xindex % 256
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (28*x3) + (784*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (28*y0) + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (28*x5) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 256.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x5 + (7168*y4)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cic2k7ifwllptfwp6vjtegpidpliczt2xa4hmacjuf7fdicvsam7.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3a2lghtpmi3b6l7yv57kuiec7mue5e5ncgoxppub6laz6xnt2ad.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((784*x1) + (200704*(y0 // 784)) + (y0 % 784)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (256*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cbablyy2jn7wrlj7rvsbgj4d7rwa327urwjwvo7vnezcx35gpv6e.py
# Source Nodes: [x_85], Original ATen: [aten.gelu, aten.gelu_backward]
# x_85 => add_26, erf_5, mul_37
triton_poi_fused_gelu_gelu_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdu44474kfmeyx2p5hamlylgxqy2cmlqcpyiztjw4z3pf4hac4n.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jljgaownmr7t5paoq5zeckuuvayqmyp6u4rldl2sn5zkxqofoq.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    x0 = xindex % 2
    x2 = (xindex // 56) % 28
    x3 = (xindex // 1568)
    x7 = xindex % 56
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r4 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x2 + (28*r4) + (3584*x7) + (200704*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coaucjq3lo2dg4hxyt6pnua6r7ak6kkbeg2c6z3pivqfpn4hdp3b.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 256)
    y0 = yindex % 28
    y1 = (yindex // 28)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (y0 + (28*x3) + (784*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (28*x5) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 256.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr1 + (x2 + (256*y0) + (7168*x3) + (200704*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4fzyrwoe3verf7y6vba54o7wqm6rf2krssbgx33vmhkuwqxot3.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((784*x0) + (200704*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp0 + tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdltsvbk7ptk7owt2i6aqc2bjfjb6gfupfbwsc2euygcj4ixo72r.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/ar/carejiuqcrof5y5tyjt6gzukfxci6fylkkcv5q6kzwokgruh77w2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/cohvvdkjov73adoaefsdgc3ec2drcxdoeh3v7zewmdsfklo7pprs.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxkneb54xm7t2ejigla2pumviq4m2gv37krdxuuxqzqpnvyhp6y.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/2w/c2wqdls2sybkez47iiik7ljohvhjrlvrsczdhfowmelly3kbuflv.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((28*x0) + (7168*((r2 + (128*x1)) % 28)) + (200704*((r2 + (128*x1)) // 784)) + (((r2 + (128*x1)) // 28) % 28)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6pulxaziw7qa7gpweutzoiot6rn5nkisng6fohab7skr2njocp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*(((r2 + (128*x1)) // 28) % 28)) + (7168*((r2 + (128*x1)) % 28)) + (200704*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rf/crfbc7gv6poconm4rgvwdm5ima7ihvfigpfpd335ksbctqy447i4.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5tpfjv42j3o2gswxujtqj2s6ngn6rlefdciqvcr66g3xsyoqfi.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_80', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 256)
    y0 = yindex % 28
    y1 = (yindex // 28)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (y0 + (28*x3) + (784*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (28*x5) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3 + (28*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 256.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x2 + (256*y0) + (7168*x3) + (200704*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yd/cydoap2lqn6rot7h2syi37da6dxva4hdodfmq4oiyelvlwwbsret.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*(((r2 + (128*x1)) // 28) % 28)) + (7168*((r2 + (128*x1)) % 28)) + (200704*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6a/c6axy4zb5vo34tfe7ibkkzsoqzn7u3v6lssidfe6alzbke2udei2.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxx6l2tjf5jtiqqretogykjbjalwr2iyb4ssahphmnqh7rniswcs.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_per_fused_add_mul_sum_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vd32aca7j7fsvffkzvfbuhfqecmu7jz4okq66u4ftnr4iuy7fx.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmze5orlx4knjtdvlcvoqnmxfspxhnxlewlx4fds6iyzr7qimzlr.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y5 = yindex
    x2 = xindex % 28
    x3 = (xindex // 28)
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x4) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x4 + (784*y5)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 + (784*y5)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 + (784*y5)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (y0 + (256*x3) + (7168*x2) + (200704*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3t3fjzaue3caz7qhcvxvlmzv3hhupba3zmsk7sl5miqdkcgafv.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    x4 = xindex % 56
    x5 = (xindex // 56) % 56
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x5 + (56*r2) + (7168*x4) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr3 + (x5 + (56*x4) + (3136*x1)), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (x5 + (56*r2) + (7168*x4) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 * tmp13
        tmp15 = 128.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp4
        tmp19 = tmp18 * tmp9
        tmp20 = tmp17 - tmp19
        tmp21 = tmp11 * tmp20
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewpcrut4jrnzc2nma4qge3que5hx7hkxpq7ib7a3fmfhtrgghki.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 128
    x2 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((3136*x1) + (401408*((r3 + (128*x0)) // 3136)) + (802816*x2) + ((r3 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((56*x1) + (7168*((r3 + (128*x0)) % 56)) + (401408*((r3 + (128*x0)) // 3136)) + (802816*x2) + (((r3 + (128*x0)) // 56) % 56)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2fgmfk52ke2v3jiurpodv2izj257hlrzjojroiou3gksdtscw6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdq4vk2cn7whdqcwvoptyp7xqyw4zqnvphsaxnsajqgkridgkues.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_89', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/ccaeb3dbwweh6krqphwwn5pjanq3ibkkvdolmcor4urmmo2rhskb.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cix5nal5fjiifgcjrjiffda5v2nmzodjlo4ujfow37rkh2tvilbq.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrx2yx45klaohyuulioiulmrtxlrm7x7wx7v273eczedgtzw6fi.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((3136*x1) + (401408*(y0 // 3136)) + (y0 % 3136)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (128*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdtetmacnrfgcymr5momt7qgvpvshd2dms745lq7hzqqlhvyeyd.py
# Source Nodes: [x_38], Original ATen: [aten.gelu, aten.gelu_backward]
# x_38 => add_12, erf_2, mul_17
triton_poi_fused_gelu_gelu_backward_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_93', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cumfapzuk6vbzdviigfkxg7q2t7lrrxuzc44iygcflqluj6eitqq.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x2 = xindex % 56
    x3 = (xindex // 56) % 56
    x4 = (xindex // 3136)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x3 + (56*r1) + (7168*x2) + (401408*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs265jjgnndw4hfgrshhbwo2fwy4esj3vxdgyegesgypgikxamjj.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 128)
    y0 = yindex % 56
    y1 = (yindex // 56)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (y0 + (56*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0 + (56*x5) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 128.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr1 + (x2 + (128*y0) + (7168*x3) + (401408*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafn3j7xxzvurwgdasaxbbw7qls5cwxqti4yljs7xyaasintecm6.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((3136*x0) + (401408*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp0 + tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtsv3ef4dupkdzpfmmhte2ezctnytkkg7ex3h6irqvxmypojs4k.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 196
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciavcht5l2bgoklcrfhtoyyw4vxnejvoxuwyncpxvgyysotrz2ws.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hj/chjfvowcu5vzgnwsdhesqi6g3nayj6mlyelkpue7pxhch2o6kmxa.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwxnhnx22vgygnifjexjl5b2o6wka67rkjsrgckeimtpqj7fbfs.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 196
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrzcybnfmhxixmnfl5cxk2zr46qoszguiltbqgrhtsya74kmpun.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((56*x0) + (7168*((r2 + (128*x1)) % 56)) + (401408*((r2 + (128*x1)) // 3136)) + (((r2 + (128*x1)) // 56) % 56)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/colgjqlc6pcrcynu6j5r3fmjppfzfbvabztuix2oopfudmcvftkn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*(((r2 + (128*x1)) // 56) % 56)) + (7168*((r2 + (128*x1)) % 56)) + (401408*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctncmuzyjaepex7buauttfwy4uvqmw7r5jvwj4cgdawb5kqxqk5d.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_103 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajumf332lffzyjvbehslacyda2z3m2dsyrkupok62q5v2uhllbz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_native_layer_norm_backward_permute_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_permute_104', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 7168
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 128)
    y0 = yindex % 56
    y1 = (yindex // 56)
    x5 = xindex
    y4 = yindex
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (y0 + (56*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x5 + (7168*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (56*x5) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3 + (56*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 128.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x2 + (128*y0) + (7168*x3) + (401408*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpaiy2enxsdqpvkkr2poj7g2k5572j4ui3e3ubift5ik2l6df6jv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*(((r2 + (128*x1)) // 56) % 56)) + (7168*((r2 + (128*x1)) % 56)) + (401408*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44jx4hmksqdv3thvsqwqld5waavijwsio2cq7qcajlgf5lzsnb5.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_red_fused_add_mul_sum_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((3136*x1) + (401408*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((3136*x1) + (401408*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirkcwsogz7gpiepg3fnks6pvmuqvm3jrkfr3lt4tmc42ash26fx.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_per_fused_add_mul_sum_107 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwj6pzpvlpcsg3pkkzog6vmuxv5gyiluudmxabmxpztmoo42vec.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_108 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cj/ccjdt6bh2eq5jcwkmcpefnmsmpb37jued24yuryo4nynmd5lu6h3.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_poi_fused_native_layer_norm_backward_109 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chff7o6ict6jjrednvfapxlw2a6uzzg2dpwxchnp524mo7dvscxt.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]

triton_red_fused_native_layer_norm_backward_permute_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_permute_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    x4 = xindex % 56
    x5 = (xindex // 56) % 56
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x5 + (56*r2) + (7168*x4) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = tl.load(in_ptr2 + (x5 + (56*x4) + (3136*x1)), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (x5 + (56*r2) + (7168*x4) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = 128.0
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12 - tmp2
        tmp15 = tmp14 * tmp7
        tmp16 = tmp13 - tmp15
        tmp17 = tmp9 * tmp16
        tl.store(out_ptr3 + (r2 + (128*x5) + (7168*x4) + (401408*x1)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/br/cbrcq34wrljw3iho4vwfr2hlvkg4z7eydjghwqyrhvb2b5it6ppb.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (802816*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr4 + ((56*x0) + (7168*(r2 % 56)) + (401408*(r2 // 3136)) + (802816*x1) + ((r2 // 56) % 56)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_121, primals_127, primals_133, primals_139, primals_141, primals_147, primals_153, primals_159, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, primals_323, primals_325, primals_331, primals_337, primals_345, mul, permute_1, mul_2, view, addmm, view_2, addmm_1, add_5, mul_8, view_5, addmm_2, view_7, addmm_3, add_9, mul_14, view_10, addmm_4, view_12, addmm_5, mul_20, permute_15, convolution_4, mul_22, view_15, addmm_6, view_17, addmm_7, add_19, mul_28, view_20, addmm_8, view_22, addmm_9, add_23, mul_34, view_25, addmm_10, view_27, addmm_11, mul_40, permute_29, convolution_8, mul_42, view_30, addmm_12, view_32, addmm_13, add_33, mul_48, view_35, addmm_14, view_37, addmm_15, add_37, mul_54, view_40, addmm_16, view_42, addmm_17, add_41, mul_60, view_45, addmm_18, view_47, addmm_19, add_45, mul_66, view_50, addmm_20, view_52, addmm_21, add_49, mul_72, view_55, addmm_22, view_57, addmm_23, add_53, mul_78, view_60, addmm_24, view_62, addmm_25, add_57, mul_84, view_65, addmm_26, view_67, addmm_27, add_61, mul_90, view_70, addmm_28, view_72, addmm_29, add_65, mul_96, view_75, addmm_30, view_77, addmm_31, add_69, mul_102, view_80, addmm_32, view_82, addmm_33, add_73, mul_108, view_85, addmm_34, view_87, addmm_35, add_77, mul_114, view_90, addmm_36, view_92, addmm_37, add_81, mul_120, view_95, addmm_38, view_97, addmm_39, add_85, mul_126, view_100, addmm_40, view_102, addmm_41, add_89, mul_132, view_105, addmm_42, view_107, addmm_43, add_93, mul_138, view_110, addmm_44, view_112, addmm_45, add_97, mul_144, view_115, addmm_46, view_117, addmm_47, add_101, mul_150, view_120, addmm_48, view_122, addmm_49, add_105, mul_156, view_125, addmm_50, view_127, addmm_51, add_109, mul_162, view_130, addmm_52, view_132, addmm_53, add_113, mul_168, view_135, addmm_54, view_137, addmm_55, add_117, mul_174, view_140, addmm_56, view_142, addmm_57, add_121, mul_180, view_145, addmm_58, view_147, addmm_59, add_125, mul_186, view_150, addmm_60, view_152, addmm_61, add_129, mul_192, view_155, addmm_62, view_157, addmm_63, add_133, mul_198, view_160, addmm_64, view_162, addmm_65, mul_204, permute_139, convolution_36, mul_206, view_165, addmm_66, view_167, addmm_67, add_143, mul_212, view_170, addmm_68, view_172, addmm_69, add_147, mul_218, view_175, addmm_70, view_177, addmm_71, mul_224, clone_109, permute_155, div, permute_162, permute_166, div_2, permute_172, permute_176, div_3, permute_182, permute_186, div_4, div_5, permute_194, permute_198, div_6, permute_204, permute_208, div_7, permute_214, permute_218, div_8, permute_224, permute_228, div_9, permute_234, permute_238, div_10, permute_244, permute_248, div_11, permute_254, permute_258, div_12, permute_264, permute_268, div_13, permute_274, permute_278, div_14, permute_284, permute_288, div_15, permute_294, permute_298, div_16, permute_304, permute_308, div_17, permute_314, permute_318, div_18, permute_324, permute_328, div_19, permute_334, permute_338, div_20, permute_344, permute_348, div_21, permute_354, permute_358, div_22, permute_364, permute_368, div_23, permute_374, permute_378, div_24, permute_384, permute_388, div_25, permute_394, permute_398, div_26, permute_404, permute_408, div_27, permute_414, permute_418, div_28, permute_424, permute_428, div_29, permute_434, permute_438, div_30, permute_444, permute_448, div_31, permute_454, permute_458, div_32, div_33, permute_466, permute_470, div_34, permute_476, permute_480, div_35, permute_486, permute_490, div_36, div_37, permute_498, permute_502, div_38, permute_508, permute_512, div_39, permute_518, permute_522, div_40, div_41, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_119, (128, 3, 4, 4), (48, 1, 12, 3))
    assert_size_stride(primals_121, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_127, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_133, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_139, (256, 128, 2, 2), (512, 1, 256, 128))
    assert_size_stride(primals_141, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_147, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_153, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_159, (512, 256, 2, 2), (1024, 1, 512, 256))
    assert_size_stride(primals_161, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_167, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_173, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_179, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_185, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_191, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_197, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_203, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_209, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_215, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_221, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_227, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_233, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_239, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_245, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_251, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_257, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_263, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_269, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_275, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_281, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_287, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_293, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_299, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_305, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_311, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_317, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_323, (1024, 512, 2, 2), (2048, 1, 1024, 512))
    assert_size_stride(primals_325, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_331, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_337, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_345, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 56, 56, 128), (401408, 1, 7168, 56))
    assert_size_stride(permute_1, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(mul_2, (8, 56, 56, 128), (401408, 1, 7168, 56))
    assert_size_stride(view, (25088, 128), (128, 1))
    assert_size_stride(addmm, (25088, 512), (512, 1))
    assert_size_stride(view_2, (25088, 512), (512, 1))
    assert_size_stride(addmm_1, (25088, 128), (128, 1))
    assert_size_stride(add_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(mul_8, (8, 56, 56, 128), (401408, 1, 7168, 56))
    assert_size_stride(view_5, (25088, 128), (128, 1))
    assert_size_stride(addmm_2, (25088, 512), (512, 1))
    assert_size_stride(view_7, (25088, 512), (512, 1))
    assert_size_stride(addmm_3, (25088, 128), (128, 1))
    assert_size_stride(add_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(mul_14, (8, 56, 56, 128), (401408, 1, 7168, 56))
    assert_size_stride(view_10, (25088, 128), (128, 1))
    assert_size_stride(addmm_4, (25088, 512), (512, 1))
    assert_size_stride(view_12, (25088, 512), (512, 1))
    assert_size_stride(addmm_5, (25088, 128), (128, 1))
    assert_size_stride(mul_20, (8, 56, 56, 128), (401408, 1, 7168, 56))
    assert_size_stride(permute_15, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_4, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(mul_22, (8, 28, 28, 256), (200704, 1, 7168, 28))
    assert_size_stride(view_15, (6272, 256), (256, 1))
    assert_size_stride(addmm_6, (6272, 1024), (1024, 1))
    assert_size_stride(view_17, (6272, 1024), (1024, 1))
    assert_size_stride(addmm_7, (6272, 256), (256, 1))
    assert_size_stride(add_19, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(mul_28, (8, 28, 28, 256), (200704, 1, 7168, 28))
    assert_size_stride(view_20, (6272, 256), (256, 1))
    assert_size_stride(addmm_8, (6272, 1024), (1024, 1))
    assert_size_stride(view_22, (6272, 1024), (1024, 1))
    assert_size_stride(addmm_9, (6272, 256), (256, 1))
    assert_size_stride(add_23, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(mul_34, (8, 28, 28, 256), (200704, 1, 7168, 28))
    assert_size_stride(view_25, (6272, 256), (256, 1))
    assert_size_stride(addmm_10, (6272, 1024), (1024, 1))
    assert_size_stride(view_27, (6272, 1024), (1024, 1))
    assert_size_stride(addmm_11, (6272, 256), (256, 1))
    assert_size_stride(mul_40, (8, 28, 28, 256), (200704, 1, 7168, 28))
    assert_size_stride(permute_29, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_8, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_42, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_30, (1568, 512), (512, 1))
    assert_size_stride(addmm_12, (1568, 2048), (2048, 1))
    assert_size_stride(view_32, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_13, (1568, 512), (512, 1))
    assert_size_stride(add_33, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_48, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_35, (1568, 512), (512, 1))
    assert_size_stride(addmm_14, (1568, 2048), (2048, 1))
    assert_size_stride(view_37, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_15, (1568, 512), (512, 1))
    assert_size_stride(add_37, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_54, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_40, (1568, 512), (512, 1))
    assert_size_stride(addmm_16, (1568, 2048), (2048, 1))
    assert_size_stride(view_42, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_17, (1568, 512), (512, 1))
    assert_size_stride(add_41, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_60, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_45, (1568, 512), (512, 1))
    assert_size_stride(addmm_18, (1568, 2048), (2048, 1))
    assert_size_stride(view_47, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_19, (1568, 512), (512, 1))
    assert_size_stride(add_45, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_66, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_50, (1568, 512), (512, 1))
    assert_size_stride(addmm_20, (1568, 2048), (2048, 1))
    assert_size_stride(view_52, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_21, (1568, 512), (512, 1))
    assert_size_stride(add_49, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_72, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_55, (1568, 512), (512, 1))
    assert_size_stride(addmm_22, (1568, 2048), (2048, 1))
    assert_size_stride(view_57, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_23, (1568, 512), (512, 1))
    assert_size_stride(add_53, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_78, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_60, (1568, 512), (512, 1))
    assert_size_stride(addmm_24, (1568, 2048), (2048, 1))
    assert_size_stride(view_62, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_25, (1568, 512), (512, 1))
    assert_size_stride(add_57, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_84, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_65, (1568, 512), (512, 1))
    assert_size_stride(addmm_26, (1568, 2048), (2048, 1))
    assert_size_stride(view_67, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_27, (1568, 512), (512, 1))
    assert_size_stride(add_61, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_90, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_70, (1568, 512), (512, 1))
    assert_size_stride(addmm_28, (1568, 2048), (2048, 1))
    assert_size_stride(view_72, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_29, (1568, 512), (512, 1))
    assert_size_stride(add_65, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_96, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_75, (1568, 512), (512, 1))
    assert_size_stride(addmm_30, (1568, 2048), (2048, 1))
    assert_size_stride(view_77, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_31, (1568, 512), (512, 1))
    assert_size_stride(add_69, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_102, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_80, (1568, 512), (512, 1))
    assert_size_stride(addmm_32, (1568, 2048), (2048, 1))
    assert_size_stride(view_82, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_33, (1568, 512), (512, 1))
    assert_size_stride(add_73, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_108, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_85, (1568, 512), (512, 1))
    assert_size_stride(addmm_34, (1568, 2048), (2048, 1))
    assert_size_stride(view_87, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_35, (1568, 512), (512, 1))
    assert_size_stride(add_77, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_114, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_90, (1568, 512), (512, 1))
    assert_size_stride(addmm_36, (1568, 2048), (2048, 1))
    assert_size_stride(view_92, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_37, (1568, 512), (512, 1))
    assert_size_stride(add_81, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_120, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_95, (1568, 512), (512, 1))
    assert_size_stride(addmm_38, (1568, 2048), (2048, 1))
    assert_size_stride(view_97, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_39, (1568, 512), (512, 1))
    assert_size_stride(add_85, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_126, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_100, (1568, 512), (512, 1))
    assert_size_stride(addmm_40, (1568, 2048), (2048, 1))
    assert_size_stride(view_102, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_41, (1568, 512), (512, 1))
    assert_size_stride(add_89, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_132, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_105, (1568, 512), (512, 1))
    assert_size_stride(addmm_42, (1568, 2048), (2048, 1))
    assert_size_stride(view_107, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_43, (1568, 512), (512, 1))
    assert_size_stride(add_93, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_138, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_110, (1568, 512), (512, 1))
    assert_size_stride(addmm_44, (1568, 2048), (2048, 1))
    assert_size_stride(view_112, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_45, (1568, 512), (512, 1))
    assert_size_stride(add_97, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_144, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_115, (1568, 512), (512, 1))
    assert_size_stride(addmm_46, (1568, 2048), (2048, 1))
    assert_size_stride(view_117, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_47, (1568, 512), (512, 1))
    assert_size_stride(add_101, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_150, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_120, (1568, 512), (512, 1))
    assert_size_stride(addmm_48, (1568, 2048), (2048, 1))
    assert_size_stride(view_122, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_49, (1568, 512), (512, 1))
    assert_size_stride(add_105, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_156, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_125, (1568, 512), (512, 1))
    assert_size_stride(addmm_50, (1568, 2048), (2048, 1))
    assert_size_stride(view_127, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_51, (1568, 512), (512, 1))
    assert_size_stride(add_109, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_162, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_130, (1568, 512), (512, 1))
    assert_size_stride(addmm_52, (1568, 2048), (2048, 1))
    assert_size_stride(view_132, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_53, (1568, 512), (512, 1))
    assert_size_stride(add_113, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_168, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_135, (1568, 512), (512, 1))
    assert_size_stride(addmm_54, (1568, 2048), (2048, 1))
    assert_size_stride(view_137, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_55, (1568, 512), (512, 1))
    assert_size_stride(add_117, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_174, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_140, (1568, 512), (512, 1))
    assert_size_stride(addmm_56, (1568, 2048), (2048, 1))
    assert_size_stride(view_142, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_57, (1568, 512), (512, 1))
    assert_size_stride(add_121, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_180, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_145, (1568, 512), (512, 1))
    assert_size_stride(addmm_58, (1568, 2048), (2048, 1))
    assert_size_stride(view_147, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_59, (1568, 512), (512, 1))
    assert_size_stride(add_125, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_186, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_150, (1568, 512), (512, 1))
    assert_size_stride(addmm_60, (1568, 2048), (2048, 1))
    assert_size_stride(view_152, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_61, (1568, 512), (512, 1))
    assert_size_stride(add_129, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_192, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_155, (1568, 512), (512, 1))
    assert_size_stride(addmm_62, (1568, 2048), (2048, 1))
    assert_size_stride(view_157, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_63, (1568, 512), (512, 1))
    assert_size_stride(add_133, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_198, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(view_160, (1568, 512), (512, 1))
    assert_size_stride(addmm_64, (1568, 2048), (2048, 1))
    assert_size_stride(view_162, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_65, (1568, 512), (512, 1))
    assert_size_stride(mul_204, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(permute_139, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_36, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(mul_206, (8, 7, 7, 1024), (50176, 1, 7168, 7))
    assert_size_stride(view_165, (392, 1024), (1024, 1))
    assert_size_stride(addmm_66, (392, 4096), (4096, 1))
    assert_size_stride(view_167, (392, 4096), (4096, 1))
    assert_size_stride(addmm_67, (392, 1024), (1024, 1))
    assert_size_stride(add_143, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(mul_212, (8, 7, 7, 1024), (50176, 1, 7168, 7))
    assert_size_stride(view_170, (392, 1024), (1024, 1))
    assert_size_stride(addmm_68, (392, 4096), (4096, 1))
    assert_size_stride(view_172, (392, 4096), (4096, 1))
    assert_size_stride(addmm_69, (392, 1024), (1024, 1))
    assert_size_stride(add_147, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(mul_218, (8, 7, 7, 1024), (50176, 1, 7168, 7))
    assert_size_stride(view_175, (392, 1024), (1024, 1))
    assert_size_stride(addmm_70, (392, 4096), (4096, 1))
    assert_size_stride(view_177, (392, 4096), (4096, 1))
    assert_size_stride(addmm_71, (392, 1024), (1024, 1))
    assert_size_stride(mul_224, (8, 1, 1, 1024), (1024, 1, 1024, 1))
    assert_size_stride(clone_109, (8, 1024), (1024, 1))
    assert_size_stride(permute_155, (1000, 1024), (1024, 1))
    assert_size_stride(div, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(permute_162, (1024, 4096), (4096, 1))
    assert_size_stride(permute_166, (4096, 1024), (1024, 1))
    assert_size_stride(div_2, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(permute_172, (1024, 4096), (4096, 1))
    assert_size_stride(permute_176, (4096, 1024), (1024, 1))
    assert_size_stride(div_3, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(permute_182, (1024, 4096), (4096, 1))
    assert_size_stride(permute_186, (4096, 1024), (1024, 1))
    assert_size_stride(div_4, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(div_5, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_194, (512, 2048), (2048, 1))
    assert_size_stride(permute_198, (2048, 512), (512, 1))
    assert_size_stride(div_6, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_204, (512, 2048), (2048, 1))
    assert_size_stride(permute_208, (2048, 512), (512, 1))
    assert_size_stride(div_7, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_214, (512, 2048), (2048, 1))
    assert_size_stride(permute_218, (2048, 512), (512, 1))
    assert_size_stride(div_8, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_224, (512, 2048), (2048, 1))
    assert_size_stride(permute_228, (2048, 512), (512, 1))
    assert_size_stride(div_9, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_234, (512, 2048), (2048, 1))
    assert_size_stride(permute_238, (2048, 512), (512, 1))
    assert_size_stride(div_10, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_244, (512, 2048), (2048, 1))
    assert_size_stride(permute_248, (2048, 512), (512, 1))
    assert_size_stride(div_11, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_254, (512, 2048), (2048, 1))
    assert_size_stride(permute_258, (2048, 512), (512, 1))
    assert_size_stride(div_12, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_264, (512, 2048), (2048, 1))
    assert_size_stride(permute_268, (2048, 512), (512, 1))
    assert_size_stride(div_13, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_274, (512, 2048), (2048, 1))
    assert_size_stride(permute_278, (2048, 512), (512, 1))
    assert_size_stride(div_14, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_284, (512, 2048), (2048, 1))
    assert_size_stride(permute_288, (2048, 512), (512, 1))
    assert_size_stride(div_15, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_294, (512, 2048), (2048, 1))
    assert_size_stride(permute_298, (2048, 512), (512, 1))
    assert_size_stride(div_16, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_304, (512, 2048), (2048, 1))
    assert_size_stride(permute_308, (2048, 512), (512, 1))
    assert_size_stride(div_17, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_314, (512, 2048), (2048, 1))
    assert_size_stride(permute_318, (2048, 512), (512, 1))
    assert_size_stride(div_18, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_324, (512, 2048), (2048, 1))
    assert_size_stride(permute_328, (2048, 512), (512, 1))
    assert_size_stride(div_19, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_334, (512, 2048), (2048, 1))
    assert_size_stride(permute_338, (2048, 512), (512, 1))
    assert_size_stride(div_20, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_344, (512, 2048), (2048, 1))
    assert_size_stride(permute_348, (2048, 512), (512, 1))
    assert_size_stride(div_21, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_354, (512, 2048), (2048, 1))
    assert_size_stride(permute_358, (2048, 512), (512, 1))
    assert_size_stride(div_22, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_364, (512, 2048), (2048, 1))
    assert_size_stride(permute_368, (2048, 512), (512, 1))
    assert_size_stride(div_23, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_374, (512, 2048), (2048, 1))
    assert_size_stride(permute_378, (2048, 512), (512, 1))
    assert_size_stride(div_24, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_384, (512, 2048), (2048, 1))
    assert_size_stride(permute_388, (2048, 512), (512, 1))
    assert_size_stride(div_25, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_394, (512, 2048), (2048, 1))
    assert_size_stride(permute_398, (2048, 512), (512, 1))
    assert_size_stride(div_26, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_404, (512, 2048), (2048, 1))
    assert_size_stride(permute_408, (2048, 512), (512, 1))
    assert_size_stride(div_27, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_414, (512, 2048), (2048, 1))
    assert_size_stride(permute_418, (2048, 512), (512, 1))
    assert_size_stride(div_28, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_424, (512, 2048), (2048, 1))
    assert_size_stride(permute_428, (2048, 512), (512, 1))
    assert_size_stride(div_29, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_434, (512, 2048), (2048, 1))
    assert_size_stride(permute_438, (2048, 512), (512, 1))
    assert_size_stride(div_30, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_444, (512, 2048), (2048, 1))
    assert_size_stride(permute_448, (2048, 512), (512, 1))
    assert_size_stride(div_31, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_454, (512, 2048), (2048, 1))
    assert_size_stride(permute_458, (2048, 512), (512, 1))
    assert_size_stride(div_32, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(div_33, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(permute_466, (256, 1024), (1024, 1))
    assert_size_stride(permute_470, (1024, 256), (256, 1))
    assert_size_stride(div_34, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(permute_476, (256, 1024), (1024, 1))
    assert_size_stride(permute_480, (1024, 256), (256, 1))
    assert_size_stride(div_35, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(permute_486, (256, 1024), (1024, 1))
    assert_size_stride(permute_490, (1024, 256), (256, 1))
    assert_size_stride(div_36, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(div_37, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(permute_498, (128, 512), (512, 1))
    assert_size_stride(permute_502, (512, 128), (128, 1))
    assert_size_stride(div_38, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(permute_508, (128, 512), (512, 1))
    assert_size_stride(permute_512, (512, 128), (128, 1))
    assert_size_stride(div_39, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(permute_518, (128, 512), (512, 1))
    assert_size_stride(permute_522, (512, 128), (128, 1))
    assert_size_stride(div_40, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(div_41, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_155, out=buf0)
        del permute_155
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_109, out=buf1)
        del clone_109
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf9 = empty((8192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_1.run(buf9, 8192, grid=grid(8192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.native_layer_norm_backward, aten.squeeze]
        triton_per_fused_as_strided_scatter_native_layer_norm_backward_squeeze_2.run(buf0, primals_117, mul_224, div, buf9, 8, 1024, grid=grid(8), stream=stream0)
        del div
        del primals_117
        buf5 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf6 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf0, mul_224, buf5, buf6, 1024, 8, grid=grid(1024), stream=stream0)
        del buf0
        del mul_224
        buf13 = empty((8, 7, 7, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf9, primals_116, buf13, 401408, grid=grid(401408), stream=stream0)
        del primals_116
        buf14 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (392, 1024), (1024, 1), 0), permute_162, out=buf14)
        del permute_162
        buf18 = reinterpret_tensor(buf14, (8, 7, 7, 4096), (200704, 28672, 4096, 1), 0); del buf14  # reuse
        # Source Nodes: [x_515], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf18, addmm_70, 1605632, grid=grid(1605632), stream=stream0)
        del addmm_70
        buf19 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (392, 4096), (4096, 1), 0), permute_166, out=buf19)
        del permute_166
        buf23 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf19, primals_114, buf23, 392, 1024, grid=grid(392), stream=stream0)
        buf24 = empty_strided((8, 7, 7, 1, 8), (392, 56, 8, 3136, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_7.run(buf19, primals_114, mul_218, buf24, 3136, 128, grid=grid(3136), stream=stream0)
        buf25 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_8.run(buf24, buf25, 392, 8, grid=grid(392), stream=stream0)
        buf31 = empty_strided((8, 1024, 7, 7), (50176, 1, 1024, 7168), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_9.run(div_2, buf19, primals_114, buf23, mul_218, buf25, buf31, 56, 7168, grid=grid(56, 7168), stream=stream0)
        del div_2
        del primals_114
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf34 = aten.convolution_backward(buf31, add_147, primals_337, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, False])
        del add_147
        del primals_337
        buf35 = buf34[0]
        buf11 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_10.run(buf9, addmm_71, buf35, addmm_69, buf11, buf37, 4096, 98, grid=grid(4096), stream=stream0)
        del addmm_69
        del addmm_71
        buf12 = empty((1, 1024, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_11.run(buf11, buf12, 1024, 4, grid=grid(1024), stream=stream0)
        buf15 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1024, 392), (1, 1024), 0), view_177, out=buf15)
        del view_177
        buf16 = reinterpret_tensor(buf11, (1, 1024, 4), (4096, 1, 1024), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf13, buf16, 4096, 98, grid=grid(4096), stream=stream0)
        del buf13
        buf17 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_sum_11.run(buf16, buf17, 1024, 4, grid=grid(1024), stream=stream0)
        buf20 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (4096, 392), (1, 4096), 0), view_175, out=buf20)
        del view_175
        buf21 = empty_strided((1, 4096, 4), (16384, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf18, buf21, 16384, 98, grid=grid(16384), stream=stream0)
        buf22 = reinterpret_tensor(buf16, (1, 4096), (4096, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf21, buf22, 4096, 4, grid=grid(4096), stream=stream0)
        buf26 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        buf28 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_15.run(buf19, mul_218, buf26, buf28, 4096, 98, grid=grid(4096), stream=stream0)
        del mul_218
        buf27 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_sum_11.run(buf26, buf27, 1024, 4, grid=grid(1024), stream=stream0)
        buf29 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_sum_11.run(buf28, buf29, 1024, 4, grid=grid(1024), stream=stream0)
        buf32 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_16.run(buf31, buf32, 4096, 98, grid=grid(4096), stream=stream0)
        buf33 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_mul_sum_11.run(buf32, buf33, 1024, 4, grid=grid(1024), stream=stream0)
        buf36 = buf34[1]
        del buf34
        buf38 = empty((1, 1024, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_11.run(buf37, buf38, 1024, 4, grid=grid(1024), stream=stream0)
        buf39 = reinterpret_tensor(buf31, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf9, buf35, primals_113, buf39, 392, 1024, grid=grid(392, 1024), stream=stream0)
        del primals_113
        buf40 = reinterpret_tensor(buf18, (392, 4096), (4096, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (392, 1024), (1024, 1), 0), permute_172, out=buf40)
        del permute_172
        buf41 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1024, 392), (1, 1024), 0), view_172, out=buf41)
        del view_172
        buf42 = reinterpret_tensor(buf37, (1, 1024, 4), (4096, 1, 1024), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf39, buf42, 4096, 98, grid=grid(4096), stream=stream0)
        buf43 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_sum_11.run(buf42, buf43, 1024, 4, grid=grid(1024), stream=stream0)
        buf44 = reinterpret_tensor(buf40, (8, 7, 7, 4096), (200704, 28672, 4096, 1), 0); del buf40  # reuse
        # Source Nodes: [x_501], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf44, addmm_68, 1605632, grid=grid(1605632), stream=stream0)
        del addmm_68
        buf45 = reinterpret_tensor(buf39, (392, 1024), (1024, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (392, 4096), (4096, 1), 0), permute_176, out=buf45)
        del permute_176
        buf46 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (4096, 392), (1, 4096), 0), view_170, out=buf46)
        del view_170
        buf47 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf44, buf47, 16384, 98, grid=grid(16384), stream=stream0)
        buf48 = reinterpret_tensor(buf42, (1, 4096), (4096, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf47, buf48, 4096, 4, grid=grid(4096), stream=stream0)
        buf49 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf45, primals_111, buf49, 392, 1024, grid=grid(392), stream=stream0)
        buf50 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_7.run(buf45, primals_111, mul_212, buf50, 3136, 128, grid=grid(3136), stream=stream0)
        buf51 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_8.run(buf50, buf51, 392, 8, grid=grid(392), stream=stream0)
        buf52 = buf32; del buf32  # reuse
        buf54 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_15.run(buf45, mul_212, buf52, buf54, 4096, 98, grid=grid(4096), stream=stream0)
        buf53 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_sum_11.run(buf52, buf53, 1024, 4, grid=grid(1024), stream=stream0)
        buf55 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_sum_11.run(buf54, buf55, 1024, 4, grid=grid(1024), stream=stream0)
        buf56 = reinterpret_tensor(buf45, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf45  # reuse
        buf57 = reinterpret_tensor(buf19, (8, 1024, 7, 7), (50176, 1, 1024, 7168), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_18.run(buf56, div_3, primals_111, buf49, mul_212, buf51, buf57, 56, 7168, grid=grid(56, 7168), stream=stream0)
        del div_3
        del mul_212
        del primals_111
        buf58 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf57, buf58, 4096, 98, grid=grid(4096), stream=stream0)
        buf59 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_mul_sum_11.run(buf58, buf59, 1024, 4, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf60 = aten.convolution_backward(buf57, add_143, primals_331, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, False])
        del add_143
        del primals_331
        buf61 = buf60[0]
        buf62 = buf60[1]
        del buf60
        buf63 = empty((1, 1024, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_per_fused_add_div_mul_sum_20.run(buf9, buf35, buf61, addmm_67, buf63, 1024, 392, grid=grid(1024), stream=stream0)
        del addmm_67
        buf64 = reinterpret_tensor(buf57, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf9, buf35, buf61, primals_110, buf64, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del primals_110
        buf65 = reinterpret_tensor(buf44, (392, 4096), (4096, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (392, 1024), (1024, 1), 0), permute_182, out=buf65)
        del permute_182
        buf66 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (1024, 392), (1, 1024), 0), view_167, out=buf66)
        del view_167
        buf67 = reinterpret_tensor(buf58, (1, 1024, 4), (4096, 1, 1024), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf64, buf67, 4096, 98, grid=grid(4096), stream=stream0)
        buf68 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_sum_11.run(buf67, buf68, 1024, 4, grid=grid(1024), stream=stream0)
        buf69 = reinterpret_tensor(buf65, (8, 7, 7, 4096), (200704, 28672, 4096, 1), 0); del buf65  # reuse
        # Source Nodes: [x_487], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_5.run(buf69, addmm_66, 1605632, grid=grid(1605632), stream=stream0)
        del addmm_66
        buf70 = reinterpret_tensor(buf64, (392, 1024), (1024, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (392, 4096), (4096, 1), 0), permute_186, out=buf70)
        del permute_186
        buf71 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (4096, 392), (1, 4096), 0), view_165, out=buf71)
        del view_165
        buf72 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf69, buf72, 16384, 98, grid=grid(16384), stream=stream0)
        buf73 = reinterpret_tensor(buf67, (1, 4096), (4096, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf72, buf73, 4096, 4, grid=grid(4096), stream=stream0)
        del buf72
        buf74 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf70, primals_108, buf74, 392, 1024, grid=grid(392), stream=stream0)
        buf75 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_7.run(buf70, primals_108, mul_206, buf75, 3136, 128, grid=grid(3136), stream=stream0)
        buf76 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_8.run(buf75, buf76, 392, 8, grid=grid(392), stream=stream0)
        del buf75
        buf77 = buf52; del buf52  # reuse
        buf79 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_15.run(buf70, mul_206, buf77, buf79, 4096, 98, grid=grid(4096), stream=stream0)
        buf78 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_sum_11.run(buf77, buf78, 1024, 4, grid=grid(1024), stream=stream0)
        del buf77
        buf80 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_sum_11.run(buf79, buf80, 1024, 4, grid=grid(1024), stream=stream0)
        buf81 = reinterpret_tensor(buf70, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf70  # reuse
        buf82 = reinterpret_tensor(buf56, (8, 1024, 7, 7), (50176, 1, 1024, 7168), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_18.run(buf81, div_4, primals_108, buf74, mul_206, buf76, buf82, 56, 7168, grid=grid(56, 7168), stream=stream0)
        del buf74
        del buf76
        del buf81
        del div_4
        del mul_206
        del primals_108
        buf83 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_19.run(buf82, buf83, 4096, 98, grid=grid(4096), stream=stream0)
        buf84 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_mul_sum_11.run(buf83, buf84, 1024, 4, grid=grid(1024), stream=stream0)
        del buf83
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf85 = aten.convolution_backward(buf82, convolution_36, primals_325, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, False])
        del buf82
        del convolution_36
        del primals_325
        buf86 = buf85[0]
        buf87 = buf85[1]
        del buf85
        buf88 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_22.run(buf88, buf9, buf61, buf86, 401408, grid=grid(401408), stream=stream0)
        del buf61
        del buf86
        del buf9
        buf89 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf88, buf89, 1024, 392, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf90 = aten.convolution_backward(buf88, permute_139, primals_323, [1024], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf88
        del permute_139
        del primals_323
        buf91 = buf90[0]
        buf92 = buf90[1]
        del buf90
        buf93 = empty_strided((8, 14, 14, 1, 4), (784, 14, 1, 6272, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_24.run(buf91, primals_106, buf93, 6272, 128, grid=grid(6272), stream=stream0)
        buf94 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_25.run(buf93, buf94, 1568, 4, grid=grid(1568), stream=stream0)
        buf95 = reinterpret_tensor(buf93, (8, 14, 14, 1, 4), (784, 56, 4, 6272, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf91, primals_106, mul_204, buf95, 6272, 128, grid=grid(6272), stream=stream0)
        buf96 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf95, buf96, 1568, 4, grid=grid(1568), stream=stream0)
        buf97 = empty((512, ), device='cuda', dtype=torch.float32)
        buf98 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_28.run(buf91, mul_204, buf97, buf98, 512, 1568, grid=grid(512), stream=stream0)
        buf99 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_backward_29.run(div_5, buf91, primals_106, buf94, mul_204, buf96, buf99, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del div_5
        del mul_204
        del primals_106
        buf102 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_30.run(buf99, primals_105, buf102, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del primals_105
        buf103 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf102, buf103, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf104 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf103, permute_194, out=buf104)
        del permute_194
        buf108 = reinterpret_tensor(buf104, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf104  # reuse
        # Source Nodes: [x_468], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf108, addmm_64, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_64
        buf109 = reinterpret_tensor(buf102, (1568, 512), (512, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (1568, 2048), (2048, 1), 0), permute_198, out=buf109)
        del permute_198
        buf113 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf109, primals_103, buf113, 1568, 512, grid=grid(1568), stream=stream0)
        buf114 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf109, primals_103, mul_198, buf114, 6272, 128, grid=grid(6272), stream=stream0)
        buf115 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf114, buf115, 1568, 4, grid=grid(1568), stream=stream0)
        buf121 = empty_strided((8, 512, 14, 14), (100352, 1, 512, 7168), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_6, buf109, primals_103, buf113, mul_198, buf115, buf121, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_6
        del primals_103
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf124 = aten.convolution_backward(buf121, add_133, primals_317, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_133
        del primals_317
        buf125 = buf124[0]
        buf100 = empty_strided((1, 512, 1, 1, 13), (6656, 1, 6656, 6656, 512), device='cuda', dtype=torch.float32)
        buf127 = empty_strided((1, 512, 1, 1, 13), (6656, 1, 6656, 6656, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_36.run(buf99, addmm_65, buf125, addmm_63, buf100, buf127, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_63
        del addmm_65
        buf101 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_37.run(buf100, buf101, 512, 13, grid=grid(512), stream=stream0)
        buf105 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (512, 1568), (1, 512), 0), view_162, out=buf105)
        del view_162
        buf106 = reinterpret_tensor(buf100, (1, 512, 13), (6656, 1, 512), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf103, buf106, 6656, 121, grid=grid(6656), stream=stream0)
        del buf103
        buf107 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf106, buf107, 512, 13, grid=grid(512), stream=stream0)
        buf110 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (2048, 1568), (1, 2048), 0), view_160, out=buf110)
        del view_160
        buf111 = empty_strided((1, 2048, 13), (26624, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf108, buf111, 26624, 121, grid=grid(26624), stream=stream0)
        buf112 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf111, buf112, 2048, 13, grid=grid(2048), stream=stream0)
        buf116 = reinterpret_tensor(buf106, (512, 13), (1, 512), 0); del buf106  # reuse
        buf118 = empty_strided((512, 13), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf109, mul_198, buf116, buf118, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_198
        buf117 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf116, buf117, 512, 13, grid=grid(512), stream=stream0)
        del buf116
        buf119 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf118, buf119, 512, 13, grid=grid(512), stream=stream0)
        buf122 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf121, buf122, 6656, 121, grid=grid(6656), stream=stream0)
        buf123 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf122, buf123, 512, 13, grid=grid(512), stream=stream0)
        buf126 = buf124[1]
        del buf124
        buf128 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_37.run(buf127, buf128, 512, 13, grid=grid(512), stream=stream0)
        buf129 = reinterpret_tensor(buf121, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_43.run(buf99, buf125, primals_102, buf129, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del primals_102
        buf130 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf129, buf130, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf131 = reinterpret_tensor(buf108, (1568, 2048), (2048, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf130, permute_204, out=buf131)
        del permute_204
        buf132 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (512, 1568), (1, 512), 0), view_157, out=buf132)
        del view_157
        buf133 = reinterpret_tensor(buf127, (1, 512, 13), (6656, 1, 512), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf130, buf133, 6656, 121, grid=grid(6656), stream=stream0)
        buf134 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf133, buf134, 512, 13, grid=grid(512), stream=stream0)
        buf135 = reinterpret_tensor(buf131, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf131  # reuse
        # Source Nodes: [x_454], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf135, addmm_62, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_62
        buf136 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 2048), (2048, 1), 0), permute_208, out=buf136)
        del permute_208
        buf137 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (2048, 1568), (1, 2048), 0), view_155, out=buf137)
        del view_155
        buf138 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf135, buf138, 26624, 121, grid=grid(26624), stream=stream0)
        buf139 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf138, buf139, 2048, 13, grid=grid(2048), stream=stream0)
        buf140 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf136, primals_100, buf140, 1568, 512, grid=grid(1568), stream=stream0)
        buf141 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf136, primals_100, mul_192, buf141, 6272, 128, grid=grid(6272), stream=stream0)
        buf142 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf141, buf142, 1568, 4, grid=grid(1568), stream=stream0)
        buf143 = reinterpret_tensor(buf133, (512, 13), (1, 512), 0); del buf133  # reuse
        buf145 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf136, mul_192, buf143, buf145, 6656, 121, grid=grid(6656), stream=stream0)
        buf144 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf143, buf144, 512, 13, grid=grid(512), stream=stream0)
        buf146 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf145, buf146, 512, 13, grid=grid(512), stream=stream0)
        buf147 = reinterpret_tensor(buf136, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf136  # reuse
        buf148 = reinterpret_tensor(buf129, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf147, div_7, primals_100, buf140, mul_192, buf142, buf148, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_7
        del mul_192
        del primals_100
        buf149 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf148, buf149, 6656, 121, grid=grid(6656), stream=stream0)
        buf150 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf149, buf150, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf151 = aten.convolution_backward(buf148, add_129, primals_311, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_129
        del primals_311
        buf152 = buf151[0]
        buf153 = buf151[1]
        del buf151
        buf154 = reinterpret_tensor(buf149, (1, 512, 1, 1, 13), (6656, 13, 6656, 6656, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_46.run(buf99, buf125, buf152, addmm_61, buf154, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_61
        buf155 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_47.run(buf154, buf155, 512, 13, grid=grid(512), stream=stream0)
        buf156 = reinterpret_tensor(buf148, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_48.run(buf99, buf125, buf152, primals_99, buf156, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del primals_99
        buf157 = reinterpret_tensor(buf147, (1568, 512), (512, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf156, buf157, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf158 = reinterpret_tensor(buf135, (1568, 2048), (2048, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf157, permute_214, out=buf158)
        del permute_214
        buf159 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (512, 1568), (1, 512), 0), view_152, out=buf159)
        del view_152
        buf160 = reinterpret_tensor(buf154, (1, 512, 13), (6656, 1, 512), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf157, buf160, 6656, 121, grid=grid(6656), stream=stream0)
        buf161 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf160, buf161, 512, 13, grid=grid(512), stream=stream0)
        buf162 = reinterpret_tensor(buf158, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf158  # reuse
        # Source Nodes: [x_440], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf162, addmm_60, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_60
        buf163 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (1568, 2048), (2048, 1), 0), permute_218, out=buf163)
        del permute_218
        buf164 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (2048, 1568), (1, 2048), 0), view_150, out=buf164)
        del view_150
        buf165 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf162, buf165, 26624, 121, grid=grid(26624), stream=stream0)
        buf166 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf165, buf166, 2048, 13, grid=grid(2048), stream=stream0)
        buf167 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf163, primals_97, buf167, 1568, 512, grid=grid(1568), stream=stream0)
        buf168 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf163, primals_97, mul_186, buf168, 6272, 128, grid=grid(6272), stream=stream0)
        buf169 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf168, buf169, 1568, 4, grid=grid(1568), stream=stream0)
        buf170 = reinterpret_tensor(buf160, (512, 13), (1, 512), 0); del buf160  # reuse
        buf172 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf163, mul_186, buf170, buf172, 6656, 121, grid=grid(6656), stream=stream0)
        buf171 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf170, buf171, 512, 13, grid=grid(512), stream=stream0)
        buf173 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf172, buf173, 512, 13, grid=grid(512), stream=stream0)
        buf174 = reinterpret_tensor(buf163, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf163  # reuse
        buf175 = reinterpret_tensor(buf156, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf174, div_8, primals_97, buf167, mul_186, buf169, buf175, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_8
        del mul_186
        del primals_97
        buf176 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf175, buf176, 6656, 121, grid=grid(6656), stream=stream0)
        buf177 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf176, buf177, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf178 = aten.convolution_backward(buf175, add_125, primals_305, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_125
        del primals_305
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf181 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_49.run(buf99, buf125, buf152, buf179, addmm_59, buf181, 512, 1568, grid=grid(512), stream=stream0)
        del addmm_59
        buf182 = reinterpret_tensor(buf175, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_50.run(buf99, buf125, buf152, buf179, primals_96, buf182, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del primals_96
        buf183 = reinterpret_tensor(buf174, (1568, 512), (512, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf182, buf183, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf184 = reinterpret_tensor(buf162, (1568, 2048), (2048, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf183, permute_224, out=buf184)
        del permute_224
        buf185 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (512, 1568), (1, 512), 0), view_147, out=buf185)
        del view_147
        buf186 = reinterpret_tensor(buf176, (1, 512, 13), (6656, 1, 512), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf183, buf186, 6656, 121, grid=grid(6656), stream=stream0)
        buf187 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf186, buf187, 512, 13, grid=grid(512), stream=stream0)
        buf188 = reinterpret_tensor(buf184, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf184  # reuse
        # Source Nodes: [x_426], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf188, addmm_58, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_58
        buf189 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (1568, 2048), (2048, 1), 0), permute_228, out=buf189)
        del permute_228
        buf190 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (2048, 1568), (1, 2048), 0), view_145, out=buf190)
        del view_145
        buf191 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf188, buf191, 26624, 121, grid=grid(26624), stream=stream0)
        buf192 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf191, buf192, 2048, 13, grid=grid(2048), stream=stream0)
        buf193 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf189, primals_94, buf193, 1568, 512, grid=grid(1568), stream=stream0)
        buf194 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf189, primals_94, mul_180, buf194, 6272, 128, grid=grid(6272), stream=stream0)
        buf195 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf194, buf195, 1568, 4, grid=grid(1568), stream=stream0)
        buf196 = reinterpret_tensor(buf186, (512, 13), (1, 512), 0); del buf186  # reuse
        buf198 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf189, mul_180, buf196, buf198, 6656, 121, grid=grid(6656), stream=stream0)
        buf197 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf196, buf197, 512, 13, grid=grid(512), stream=stream0)
        buf199 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf198, buf199, 512, 13, grid=grid(512), stream=stream0)
        buf200 = reinterpret_tensor(buf189, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf189  # reuse
        buf201 = reinterpret_tensor(buf182, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf200, div_9, primals_94, buf193, mul_180, buf195, buf201, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_9
        del mul_180
        del primals_94
        buf202 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf201, buf202, 6656, 121, grid=grid(6656), stream=stream0)
        buf203 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf202, buf203, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf204 = aten.convolution_backward(buf201, add_121, primals_299, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_121
        del primals_299
        buf205 = buf204[0]
        buf206 = buf204[1]
        del buf204
        buf207 = buf125; del buf125  # reuse
        buf210 = reinterpret_tensor(buf201, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_51.run(buf207, buf99, buf152, buf179, buf205, primals_93, buf210, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del primals_93
        buf208 = reinterpret_tensor(buf202, (1, 512, 1, 1, 13), (6656, 13, 6656, 6656, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf207, addmm_57, buf208, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_57
        buf209 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_47.run(buf208, buf209, 512, 13, grid=grid(512), stream=stream0)
        buf211 = reinterpret_tensor(buf99, (1568, 512), (512, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf210, buf211, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf212 = reinterpret_tensor(buf188, (1568, 2048), (2048, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf211, permute_234, out=buf212)
        del permute_234
        buf213 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 1568), (1, 512), 0), view_142, out=buf213)
        del view_142
        buf214 = reinterpret_tensor(buf208, (1, 512, 13), (6656, 1, 512), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf211, buf214, 6656, 121, grid=grid(6656), stream=stream0)
        buf215 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf214, buf215, 512, 13, grid=grid(512), stream=stream0)
        buf216 = reinterpret_tensor(buf212, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf212  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf216, addmm_56, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_56
        buf217 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (1568, 2048), (2048, 1), 0), permute_238, out=buf217)
        del permute_238
        buf218 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (2048, 1568), (1, 2048), 0), view_140, out=buf218)
        del view_140
        buf219 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf216, buf219, 26624, 121, grid=grid(26624), stream=stream0)
        buf220 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf219, buf220, 2048, 13, grid=grid(2048), stream=stream0)
        buf221 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf217, primals_91, buf221, 1568, 512, grid=grid(1568), stream=stream0)
        buf222 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf217, primals_91, mul_174, buf222, 6272, 128, grid=grid(6272), stream=stream0)
        buf223 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf222, buf223, 1568, 4, grid=grid(1568), stream=stream0)
        buf224 = reinterpret_tensor(buf214, (512, 13), (1, 512), 0); del buf214  # reuse
        buf226 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf217, mul_174, buf224, buf226, 6656, 121, grid=grid(6656), stream=stream0)
        buf225 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf224, buf225, 512, 13, grid=grid(512), stream=stream0)
        buf227 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf226, buf227, 512, 13, grid=grid(512), stream=stream0)
        buf228 = reinterpret_tensor(buf217, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf217  # reuse
        buf229 = reinterpret_tensor(buf210, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf228, div_10, primals_91, buf221, mul_174, buf223, buf229, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_10
        del mul_174
        del primals_91
        buf230 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf229, buf230, 6656, 121, grid=grid(6656), stream=stream0)
        buf231 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf230, buf231, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf232 = aten.convolution_backward(buf229, add_117, primals_293, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_117
        del primals_293
        buf233 = buf232[0]
        buf234 = buf232[1]
        del buf232
        buf236 = reinterpret_tensor(buf229, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_53.run(buf207, buf233, primals_90, buf236, 802816, grid=grid(802816), stream=stream0)
        del primals_90
        buf237 = reinterpret_tensor(buf228, (1568, 512), (512, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf236, buf237, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf238 = reinterpret_tensor(buf216, (1568, 2048), (2048, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf237, permute_244, out=buf238)
        del permute_244
        buf242 = reinterpret_tensor(buf238, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf238  # reuse
        # Source Nodes: [x_398], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf242, addmm_54, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_54
        buf243 = reinterpret_tensor(buf236, (1568, 512), (512, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 2048), (2048, 1), 0), permute_248, out=buf243)
        del permute_248
        buf247 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf243, primals_88, buf247, 1568, 512, grid=grid(1568), stream=stream0)
        buf248 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf243, primals_88, mul_168, buf248, 6272, 128, grid=grid(6272), stream=stream0)
        buf249 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf248, buf249, 1568, 4, grid=grid(1568), stream=stream0)
        buf255 = reinterpret_tensor(buf205, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_11, buf243, primals_88, buf247, mul_168, buf249, buf255, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_11
        del primals_88
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf258 = aten.convolution_backward(buf255, add_113, primals_287, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_113
        del primals_287
        buf259 = buf258[0]
        buf262 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_54.run(buf207, buf233, buf259, primals_87, buf262, 802816, grid=grid(802816), stream=stream0)
        del primals_87
        buf263 = reinterpret_tensor(buf152, (1568, 512), (512, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf262, buf263, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf264 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf263, permute_254, out=buf264)
        del permute_254
        buf268 = reinterpret_tensor(buf264, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf264  # reuse
        # Source Nodes: [x_384], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf268, addmm_52, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_52
        buf269 = reinterpret_tensor(buf262, (1568, 512), (512, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (1568, 2048), (2048, 1), 0), permute_258, out=buf269)
        del permute_258
        buf273 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf269, primals_85, buf273, 1568, 512, grid=grid(1568), stream=stream0)
        buf274 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf269, primals_85, mul_162, buf274, 6272, 128, grid=grid(6272), stream=stream0)
        buf275 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf274, buf275, 1568, 4, grid=grid(1568), stream=stream0)
        buf281 = reinterpret_tensor(buf200, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_12, buf269, primals_85, buf273, mul_162, buf275, buf281, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_12
        del primals_85
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf284 = aten.convolution_backward(buf281, add_109, primals_281, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_109
        del primals_281
        buf285 = buf284[0]
        buf235 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf261 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf287 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_55.run(buf207, buf233, addmm_55, buf259, addmm_53, buf285, addmm_51, buf235, buf261, buf287, 512, 1568, grid=grid(512), stream=stream0)
        del addmm_51
        del addmm_53
        del addmm_55
        buf239 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 1568), (1, 512), 0), view_137, out=buf239)
        del view_137
        buf240 = reinterpret_tensor(buf230, (1, 512, 13), (6656, 1, 512), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf237, buf240, 6656, 121, grid=grid(6656), stream=stream0)
        del buf237
        buf241 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf240, buf241, 512, 13, grid=grid(512), stream=stream0)
        buf244 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (2048, 1568), (1, 2048), 0), view_135, out=buf244)
        del view_135
        buf245 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf242, buf245, 26624, 121, grid=grid(26624), stream=stream0)
        buf246 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf245, buf246, 2048, 13, grid=grid(2048), stream=stream0)
        buf250 = reinterpret_tensor(buf240, (512, 13), (1, 512), 0); del buf240  # reuse
        buf252 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf243, mul_168, buf250, buf252, 6656, 121, grid=grid(6656), stream=stream0)
        del buf243
        del mul_168
        buf251 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf250, buf251, 512, 13, grid=grid(512), stream=stream0)
        buf253 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf252, buf253, 512, 13, grid=grid(512), stream=stream0)
        buf256 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf255, buf256, 6656, 121, grid=grid(6656), stream=stream0)
        del buf255
        buf257 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf256, buf257, 512, 13, grid=grid(512), stream=stream0)
        buf260 = buf258[1]
        del buf258
        buf265 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (512, 1568), (1, 512), 0), view_132, out=buf265)
        del view_132
        buf266 = reinterpret_tensor(buf256, (1, 512, 13), (6656, 1, 512), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf263, buf266, 6656, 121, grid=grid(6656), stream=stream0)
        del buf263
        buf267 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf266, buf267, 512, 13, grid=grid(512), stream=stream0)
        buf270 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (2048, 1568), (1, 2048), 0), view_130, out=buf270)
        del view_130
        buf271 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf268, buf271, 26624, 121, grid=grid(26624), stream=stream0)
        buf272 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf271, buf272, 2048, 13, grid=grid(2048), stream=stream0)
        buf276 = reinterpret_tensor(buf266, (512, 13), (1, 512), 0); del buf266  # reuse
        buf278 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf269, mul_162, buf276, buf278, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_162
        buf277 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf276, buf277, 512, 13, grid=grid(512), stream=stream0)
        buf279 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf278, buf279, 512, 13, grid=grid(512), stream=stream0)
        buf282 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf281, buf282, 6656, 121, grid=grid(6656), stream=stream0)
        buf283 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf282, buf283, 512, 13, grid=grid(512), stream=stream0)
        buf286 = buf284[1]
        del buf284
        buf288 = reinterpret_tensor(buf281, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_56.run(buf207, buf233, buf259, buf285, primals_84, buf288, 802816, grid=grid(802816), stream=stream0)
        del primals_84
        buf289 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf288, buf289, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf290 = reinterpret_tensor(buf268, (1568, 2048), (2048, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf289, permute_264, out=buf290)
        del permute_264
        buf291 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (512, 1568), (1, 512), 0), view_127, out=buf291)
        del view_127
        buf292 = reinterpret_tensor(buf282, (1, 512, 13), (6656, 1, 512), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf289, buf292, 6656, 121, grid=grid(6656), stream=stream0)
        buf293 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf292, buf293, 512, 13, grid=grid(512), stream=stream0)
        buf294 = reinterpret_tensor(buf290, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf290  # reuse
        # Source Nodes: [x_370], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf294, addmm_50, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_50
        buf295 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf294, (1568, 2048), (2048, 1), 0), permute_268, out=buf295)
        del permute_268
        buf296 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf294, (2048, 1568), (1, 2048), 0), view_125, out=buf296)
        del view_125
        buf297 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf294, buf297, 26624, 121, grid=grid(26624), stream=stream0)
        buf298 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf297, buf298, 2048, 13, grid=grid(2048), stream=stream0)
        buf299 = buf275; del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf295, primals_82, buf299, 1568, 512, grid=grid(1568), stream=stream0)
        buf300 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf295, primals_82, mul_156, buf300, 6272, 128, grid=grid(6272), stream=stream0)
        buf301 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf300, buf301, 1568, 4, grid=grid(1568), stream=stream0)
        buf302 = reinterpret_tensor(buf292, (512, 13), (1, 512), 0); del buf292  # reuse
        buf304 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf295, mul_156, buf302, buf304, 6656, 121, grid=grid(6656), stream=stream0)
        buf303 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf302, buf303, 512, 13, grid=grid(512), stream=stream0)
        buf305 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf304, buf305, 512, 13, grid=grid(512), stream=stream0)
        buf306 = reinterpret_tensor(buf295, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf295  # reuse
        buf307 = reinterpret_tensor(buf288, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf306, div_13, primals_82, buf299, mul_156, buf301, buf307, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_13
        del mul_156
        del primals_82
        buf308 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf307, buf308, 6656, 121, grid=grid(6656), stream=stream0)
        buf309 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf308, buf309, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf310 = aten.convolution_backward(buf307, add_105, primals_275, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_105
        del primals_275
        buf311 = buf310[0]
        buf312 = buf310[1]
        del buf310
        buf313 = buf207; del buf207  # reuse
        buf316 = reinterpret_tensor(buf307, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_57.run(buf313, buf233, buf259, buf285, buf311, primals_81, buf316, 802816, grid=grid(802816), stream=stream0)
        del primals_81
        buf314 = reinterpret_tensor(buf308, (1, 512, 1, 1, 13), (6656, 13, 6656, 6656, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf313, addmm_49, buf314, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_49
        buf315 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_47.run(buf314, buf315, 512, 13, grid=grid(512), stream=stream0)
        buf317 = reinterpret_tensor(buf311, (1568, 512), (512, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf316, buf317, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf318 = reinterpret_tensor(buf294, (1568, 2048), (2048, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf317, permute_274, out=buf318)
        del permute_274
        buf319 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (512, 1568), (1, 512), 0), view_122, out=buf319)
        del view_122
        buf320 = reinterpret_tensor(buf314, (1, 512, 13), (6656, 1, 512), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf317, buf320, 6656, 121, grid=grid(6656), stream=stream0)
        buf321 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf320, buf321, 512, 13, grid=grid(512), stream=stream0)
        buf322 = reinterpret_tensor(buf318, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf318  # reuse
        # Source Nodes: [x_356], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf322, addmm_48, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_48
        buf323 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (1568, 2048), (2048, 1), 0), permute_278, out=buf323)
        del permute_278
        buf324 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (2048, 1568), (1, 2048), 0), view_120, out=buf324)
        del view_120
        buf325 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf322, buf325, 26624, 121, grid=grid(26624), stream=stream0)
        buf326 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf325, buf326, 2048, 13, grid=grid(2048), stream=stream0)
        buf327 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf323, primals_79, buf327, 1568, 512, grid=grid(1568), stream=stream0)
        buf328 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf323, primals_79, mul_150, buf328, 6272, 128, grid=grid(6272), stream=stream0)
        buf329 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf328, buf329, 1568, 4, grid=grid(1568), stream=stream0)
        buf330 = reinterpret_tensor(buf320, (512, 13), (1, 512), 0); del buf320  # reuse
        buf332 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf323, mul_150, buf330, buf332, 6656, 121, grid=grid(6656), stream=stream0)
        buf331 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf330, buf331, 512, 13, grid=grid(512), stream=stream0)
        buf333 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf332, buf333, 512, 13, grid=grid(512), stream=stream0)
        buf334 = reinterpret_tensor(buf323, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf323  # reuse
        buf335 = reinterpret_tensor(buf316, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf334, div_14, primals_79, buf327, mul_150, buf329, buf335, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_14
        del mul_150
        del primals_79
        buf336 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf335, buf336, 6656, 121, grid=grid(6656), stream=stream0)
        buf337 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf336, buf337, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf338 = aten.convolution_backward(buf335, add_101, primals_269, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_101
        del primals_269
        buf339 = buf338[0]
        buf340 = buf338[1]
        del buf338
        buf342 = reinterpret_tensor(buf335, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_53.run(buf313, buf339, primals_78, buf342, 802816, grid=grid(802816), stream=stream0)
        del primals_78
        buf343 = reinterpret_tensor(buf334, (1568, 512), (512, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf342, buf343, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf344 = reinterpret_tensor(buf322, (1568, 2048), (2048, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf343, permute_284, out=buf344)
        del permute_284
        buf348 = reinterpret_tensor(buf344, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf344  # reuse
        # Source Nodes: [x_342], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf348, addmm_46, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_46
        buf349 = reinterpret_tensor(buf342, (1568, 512), (512, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (1568, 2048), (2048, 1), 0), permute_288, out=buf349)
        del permute_288
        buf353 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf349, primals_76, buf353, 1568, 512, grid=grid(1568), stream=stream0)
        buf354 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf349, primals_76, mul_144, buf354, 6272, 128, grid=grid(6272), stream=stream0)
        buf355 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf354, buf355, 1568, 4, grid=grid(1568), stream=stream0)
        buf361 = reinterpret_tensor(buf285, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_15, buf349, primals_76, buf353, mul_144, buf355, buf361, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_15
        del primals_76
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf364 = aten.convolution_backward(buf361, add_97, primals_263, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_97
        del primals_263
        buf365 = buf364[0]
        buf368 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_54.run(buf313, buf339, buf365, primals_75, buf368, 802816, grid=grid(802816), stream=stream0)
        del primals_75
        buf369 = reinterpret_tensor(buf233, (1568, 512), (512, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf368, buf369, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf370 = reinterpret_tensor(buf242, (1568, 2048), (2048, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf369, permute_294, out=buf370)
        del permute_294
        buf374 = reinterpret_tensor(buf370, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf370  # reuse
        # Source Nodes: [x_328], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf374, addmm_44, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_44
        buf375 = reinterpret_tensor(buf368, (1568, 512), (512, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (1568, 2048), (2048, 1), 0), permute_298, out=buf375)
        del permute_298
        buf379 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf375, primals_73, buf379, 1568, 512, grid=grid(1568), stream=stream0)
        buf380 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf375, primals_73, mul_138, buf380, 6272, 128, grid=grid(6272), stream=stream0)
        buf381 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf380, buf381, 1568, 4, grid=grid(1568), stream=stream0)
        buf387 = reinterpret_tensor(buf306, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_16, buf375, primals_73, buf379, mul_138, buf381, buf387, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_16
        del primals_73
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf390 = aten.convolution_backward(buf387, add_93, primals_257, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_93
        del primals_257
        buf391 = buf390[0]
        buf341 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf367 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf393 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_55.run(buf313, buf339, addmm_47, buf365, addmm_45, buf391, addmm_43, buf341, buf367, buf393, 512, 1568, grid=grid(512), stream=stream0)
        del addmm_43
        del addmm_45
        del addmm_47
        buf345 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (512, 1568), (1, 512), 0), view_117, out=buf345)
        del view_117
        buf346 = reinterpret_tensor(buf336, (1, 512, 13), (6656, 1, 512), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf343, buf346, 6656, 121, grid=grid(6656), stream=stream0)
        del buf343
        buf347 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf346, buf347, 512, 13, grid=grid(512), stream=stream0)
        buf350 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (2048, 1568), (1, 2048), 0), view_115, out=buf350)
        del view_115
        buf351 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf348, buf351, 26624, 121, grid=grid(26624), stream=stream0)
        buf352 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf351, buf352, 2048, 13, grid=grid(2048), stream=stream0)
        buf356 = reinterpret_tensor(buf346, (512, 13), (1, 512), 0); del buf346  # reuse
        buf358 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf349, mul_144, buf356, buf358, 6656, 121, grid=grid(6656), stream=stream0)
        del buf349
        del mul_144
        buf357 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf356, buf357, 512, 13, grid=grid(512), stream=stream0)
        buf359 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf358, buf359, 512, 13, grid=grid(512), stream=stream0)
        buf362 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf361, buf362, 6656, 121, grid=grid(6656), stream=stream0)
        del buf361
        buf363 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf362, buf363, 512, 13, grid=grid(512), stream=stream0)
        buf366 = buf364[1]
        del buf364
        buf371 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (512, 1568), (1, 512), 0), view_112, out=buf371)
        del view_112
        buf372 = reinterpret_tensor(buf362, (1, 512, 13), (6656, 1, 512), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf369, buf372, 6656, 121, grid=grid(6656), stream=stream0)
        del buf369
        buf373 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf372, buf373, 512, 13, grid=grid(512), stream=stream0)
        buf376 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (2048, 1568), (1, 2048), 0), view_110, out=buf376)
        del view_110
        buf377 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf374, buf377, 26624, 121, grid=grid(26624), stream=stream0)
        buf378 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf377, buf378, 2048, 13, grid=grid(2048), stream=stream0)
        buf382 = reinterpret_tensor(buf372, (512, 13), (1, 512), 0); del buf372  # reuse
        buf384 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf375, mul_138, buf382, buf384, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_138
        buf383 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf382, buf383, 512, 13, grid=grid(512), stream=stream0)
        buf385 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf384, buf385, 512, 13, grid=grid(512), stream=stream0)
        buf388 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf387, buf388, 6656, 121, grid=grid(6656), stream=stream0)
        buf389 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf388, buf389, 512, 13, grid=grid(512), stream=stream0)
        buf392 = buf390[1]
        del buf390
        buf394 = reinterpret_tensor(buf387, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_56.run(buf313, buf339, buf365, buf391, primals_72, buf394, 802816, grid=grid(802816), stream=stream0)
        del primals_72
        buf395 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf394, buf395, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf396 = reinterpret_tensor(buf374, (1568, 2048), (2048, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf395, permute_304, out=buf396)
        del permute_304
        buf397 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (512, 1568), (1, 512), 0), view_107, out=buf397)
        del view_107
        buf398 = reinterpret_tensor(buf388, (1, 512, 13), (6656, 1, 512), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf395, buf398, 6656, 121, grid=grid(6656), stream=stream0)
        buf399 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf398, buf399, 512, 13, grid=grid(512), stream=stream0)
        buf400 = reinterpret_tensor(buf396, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf396  # reuse
        # Source Nodes: [x_314], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf400, addmm_42, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_42
        buf401 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (1568, 2048), (2048, 1), 0), permute_308, out=buf401)
        del permute_308
        buf402 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (2048, 1568), (1, 2048), 0), view_105, out=buf402)
        del view_105
        buf403 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf400, buf403, 26624, 121, grid=grid(26624), stream=stream0)
        buf404 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf403, buf404, 2048, 13, grid=grid(2048), stream=stream0)
        buf405 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf401, primals_70, buf405, 1568, 512, grid=grid(1568), stream=stream0)
        buf406 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf401, primals_70, mul_132, buf406, 6272, 128, grid=grid(6272), stream=stream0)
        buf407 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf406, buf407, 1568, 4, grid=grid(1568), stream=stream0)
        buf408 = reinterpret_tensor(buf398, (512, 13), (1, 512), 0); del buf398  # reuse
        buf410 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf401, mul_132, buf408, buf410, 6656, 121, grid=grid(6656), stream=stream0)
        buf409 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf408, buf409, 512, 13, grid=grid(512), stream=stream0)
        buf411 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf410, buf411, 512, 13, grid=grid(512), stream=stream0)
        buf412 = reinterpret_tensor(buf401, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf401  # reuse
        buf413 = reinterpret_tensor(buf394, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf412, div_17, primals_70, buf405, mul_132, buf407, buf413, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_17
        del mul_132
        del primals_70
        buf414 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf413, buf414, 6656, 121, grid=grid(6656), stream=stream0)
        buf415 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf414, buf415, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf416 = aten.convolution_backward(buf413, add_89, primals_251, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_89
        del primals_251
        buf417 = buf416[0]
        buf418 = buf416[1]
        del buf416
        buf419 = buf313; del buf313  # reuse
        buf422 = reinterpret_tensor(buf413, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_57.run(buf419, buf339, buf365, buf391, buf417, primals_69, buf422, 802816, grid=grid(802816), stream=stream0)
        del primals_69
        buf420 = reinterpret_tensor(buf414, (1, 512, 1, 1, 13), (6656, 13, 6656, 6656, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf419, addmm_41, buf420, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_41
        buf421 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_47.run(buf420, buf421, 512, 13, grid=grid(512), stream=stream0)
        buf423 = reinterpret_tensor(buf417, (1568, 512), (512, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf422, buf423, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf424 = reinterpret_tensor(buf400, (1568, 2048), (2048, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf423, permute_314, out=buf424)
        del permute_314
        buf425 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 1568), (1, 512), 0), view_102, out=buf425)
        del view_102
        buf426 = reinterpret_tensor(buf420, (1, 512, 13), (6656, 1, 512), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf423, buf426, 6656, 121, grid=grid(6656), stream=stream0)
        buf427 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf426, buf427, 512, 13, grid=grid(512), stream=stream0)
        buf428 = reinterpret_tensor(buf424, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf424  # reuse
        # Source Nodes: [x_300], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf428, addmm_40, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_40
        buf429 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1568, 2048), (2048, 1), 0), permute_318, out=buf429)
        del permute_318
        buf430 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (2048, 1568), (1, 2048), 0), view_100, out=buf430)
        del view_100
        buf431 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf428, buf431, 26624, 121, grid=grid(26624), stream=stream0)
        buf432 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf431, buf432, 2048, 13, grid=grid(2048), stream=stream0)
        buf433 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf429, primals_67, buf433, 1568, 512, grid=grid(1568), stream=stream0)
        buf434 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf429, primals_67, mul_126, buf434, 6272, 128, grid=grid(6272), stream=stream0)
        buf435 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf434, buf435, 1568, 4, grid=grid(1568), stream=stream0)
        buf436 = reinterpret_tensor(buf426, (512, 13), (1, 512), 0); del buf426  # reuse
        buf438 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf429, mul_126, buf436, buf438, 6656, 121, grid=grid(6656), stream=stream0)
        buf437 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf436, buf437, 512, 13, grid=grid(512), stream=stream0)
        buf439 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf438, buf439, 512, 13, grid=grid(512), stream=stream0)
        buf440 = reinterpret_tensor(buf429, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf429  # reuse
        buf441 = reinterpret_tensor(buf422, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf440, div_18, primals_67, buf433, mul_126, buf435, buf441, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_18
        del mul_126
        del primals_67
        buf442 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf441, buf442, 6656, 121, grid=grid(6656), stream=stream0)
        buf443 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf442, buf443, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf444 = aten.convolution_backward(buf441, add_85, primals_245, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_85
        del primals_245
        buf445 = buf444[0]
        buf446 = buf444[1]
        del buf444
        buf448 = reinterpret_tensor(buf441, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_53.run(buf419, buf445, primals_66, buf448, 802816, grid=grid(802816), stream=stream0)
        del primals_66
        buf449 = reinterpret_tensor(buf440, (1568, 512), (512, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf448, buf449, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf450 = reinterpret_tensor(buf428, (1568, 2048), (2048, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf449, permute_324, out=buf450)
        del permute_324
        buf454 = reinterpret_tensor(buf450, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf450  # reuse
        # Source Nodes: [x_286], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf454, addmm_38, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_38
        buf455 = reinterpret_tensor(buf448, (1568, 512), (512, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (1568, 2048), (2048, 1), 0), permute_328, out=buf455)
        del permute_328
        buf459 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf455, primals_64, buf459, 1568, 512, grid=grid(1568), stream=stream0)
        buf460 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf455, primals_64, mul_120, buf460, 6272, 128, grid=grid(6272), stream=stream0)
        buf461 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf460, buf461, 1568, 4, grid=grid(1568), stream=stream0)
        buf467 = reinterpret_tensor(buf391, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_19, buf455, primals_64, buf459, mul_120, buf461, buf467, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_19
        del primals_64
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf470 = aten.convolution_backward(buf467, add_81, primals_239, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_81
        del primals_239
        buf471 = buf470[0]
        buf474 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_54.run(buf419, buf445, buf471, primals_63, buf474, 802816, grid=grid(802816), stream=stream0)
        del primals_63
        buf475 = reinterpret_tensor(buf339, (1568, 512), (512, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf474, buf475, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf476 = reinterpret_tensor(buf348, (1568, 2048), (2048, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf475, permute_334, out=buf476)
        del permute_334
        buf480 = reinterpret_tensor(buf476, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf476  # reuse
        # Source Nodes: [x_272], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf480, addmm_36, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_36
        buf481 = reinterpret_tensor(buf474, (1568, 512), (512, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (1568, 2048), (2048, 1), 0), permute_338, out=buf481)
        del permute_338
        buf485 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf481, primals_61, buf485, 1568, 512, grid=grid(1568), stream=stream0)
        buf486 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf481, primals_61, mul_114, buf486, 6272, 128, grid=grid(6272), stream=stream0)
        buf487 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf486, buf487, 1568, 4, grid=grid(1568), stream=stream0)
        buf493 = reinterpret_tensor(buf412, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_20, buf481, primals_61, buf485, mul_114, buf487, buf493, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_20
        del primals_61
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf496 = aten.convolution_backward(buf493, add_77, primals_233, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_77
        del primals_233
        buf497 = buf496[0]
        buf447 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf473 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf499 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_55.run(buf419, buf445, addmm_39, buf471, addmm_37, buf497, addmm_35, buf447, buf473, buf499, 512, 1568, grid=grid(512), stream=stream0)
        del addmm_35
        del addmm_37
        del addmm_39
        buf451 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (512, 1568), (1, 512), 0), view_97, out=buf451)
        del view_97
        buf452 = reinterpret_tensor(buf442, (1, 512, 13), (6656, 1, 512), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf449, buf452, 6656, 121, grid=grid(6656), stream=stream0)
        del buf449
        buf453 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf452, buf453, 512, 13, grid=grid(512), stream=stream0)
        buf456 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (2048, 1568), (1, 2048), 0), view_95, out=buf456)
        del view_95
        buf457 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf454, buf457, 26624, 121, grid=grid(26624), stream=stream0)
        buf458 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf457, buf458, 2048, 13, grid=grid(2048), stream=stream0)
        buf462 = reinterpret_tensor(buf452, (512, 13), (1, 512), 0); del buf452  # reuse
        buf464 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf455, mul_120, buf462, buf464, 6656, 121, grid=grid(6656), stream=stream0)
        del buf455
        del mul_120
        buf463 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf462, buf463, 512, 13, grid=grid(512), stream=stream0)
        buf465 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf464, buf465, 512, 13, grid=grid(512), stream=stream0)
        buf468 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf467, buf468, 6656, 121, grid=grid(6656), stream=stream0)
        del buf467
        buf469 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf468, buf469, 512, 13, grid=grid(512), stream=stream0)
        buf472 = buf470[1]
        del buf470
        buf477 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (512, 1568), (1, 512), 0), view_92, out=buf477)
        del view_92
        buf478 = reinterpret_tensor(buf468, (1, 512, 13), (6656, 1, 512), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf475, buf478, 6656, 121, grid=grid(6656), stream=stream0)
        del buf475
        buf479 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf478, buf479, 512, 13, grid=grid(512), stream=stream0)
        buf482 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (2048, 1568), (1, 2048), 0), view_90, out=buf482)
        del view_90
        buf483 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf480, buf483, 26624, 121, grid=grid(26624), stream=stream0)
        buf484 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf483, buf484, 2048, 13, grid=grid(2048), stream=stream0)
        buf488 = reinterpret_tensor(buf478, (512, 13), (1, 512), 0); del buf478  # reuse
        buf490 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf481, mul_114, buf488, buf490, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_114
        buf489 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf488, buf489, 512, 13, grid=grid(512), stream=stream0)
        buf491 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf490, buf491, 512, 13, grid=grid(512), stream=stream0)
        buf494 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf493, buf494, 6656, 121, grid=grid(6656), stream=stream0)
        buf495 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf494, buf495, 512, 13, grid=grid(512), stream=stream0)
        buf498 = buf496[1]
        del buf496
        buf500 = reinterpret_tensor(buf493, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_56.run(buf419, buf445, buf471, buf497, primals_60, buf500, 802816, grid=grid(802816), stream=stream0)
        del primals_60
        buf501 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf500, buf501, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf502 = reinterpret_tensor(buf480, (1568, 2048), (2048, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf501, permute_344, out=buf502)
        del permute_344
        buf503 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (512, 1568), (1, 512), 0), view_87, out=buf503)
        del view_87
        buf504 = reinterpret_tensor(buf494, (1, 512, 13), (6656, 1, 512), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf501, buf504, 6656, 121, grid=grid(6656), stream=stream0)
        buf505 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf504, buf505, 512, 13, grid=grid(512), stream=stream0)
        buf506 = reinterpret_tensor(buf502, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf502  # reuse
        # Source Nodes: [x_258], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf506, addmm_34, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_34
        buf507 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (1568, 2048), (2048, 1), 0), permute_348, out=buf507)
        del permute_348
        buf508 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (2048, 1568), (1, 2048), 0), view_85, out=buf508)
        del view_85
        buf509 = buf483; del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf506, buf509, 26624, 121, grid=grid(26624), stream=stream0)
        buf510 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf509, buf510, 2048, 13, grid=grid(2048), stream=stream0)
        buf511 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf507, primals_58, buf511, 1568, 512, grid=grid(1568), stream=stream0)
        buf512 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf507, primals_58, mul_108, buf512, 6272, 128, grid=grid(6272), stream=stream0)
        buf513 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf512, buf513, 1568, 4, grid=grid(1568), stream=stream0)
        buf514 = reinterpret_tensor(buf504, (512, 13), (1, 512), 0); del buf504  # reuse
        buf516 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf507, mul_108, buf514, buf516, 6656, 121, grid=grid(6656), stream=stream0)
        buf515 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf514, buf515, 512, 13, grid=grid(512), stream=stream0)
        buf517 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf516, buf517, 512, 13, grid=grid(512), stream=stream0)
        buf518 = reinterpret_tensor(buf507, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf507  # reuse
        buf519 = reinterpret_tensor(buf500, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf518, div_21, primals_58, buf511, mul_108, buf513, buf519, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_21
        del mul_108
        del primals_58
        buf520 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf519, buf520, 6656, 121, grid=grid(6656), stream=stream0)
        buf521 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf520, buf521, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf522 = aten.convolution_backward(buf519, add_73, primals_227, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_73
        del primals_227
        buf523 = buf522[0]
        buf524 = buf522[1]
        del buf522
        buf525 = buf419; del buf419  # reuse
        buf528 = reinterpret_tensor(buf519, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_57.run(buf525, buf445, buf471, buf497, buf523, primals_57, buf528, 802816, grid=grid(802816), stream=stream0)
        del primals_57
        buf526 = reinterpret_tensor(buf520, (1, 512, 1, 1, 13), (6656, 13, 6656, 6656, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf525, addmm_33, buf526, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_33
        buf527 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_47.run(buf526, buf527, 512, 13, grid=grid(512), stream=stream0)
        buf529 = reinterpret_tensor(buf523, (1568, 512), (512, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf528, buf529, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf530 = reinterpret_tensor(buf506, (1568, 2048), (2048, 1), 0); del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf529, permute_354, out=buf530)
        del permute_354
        buf531 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (512, 1568), (1, 512), 0), view_82, out=buf531)
        del view_82
        buf532 = reinterpret_tensor(buf526, (1, 512, 13), (6656, 1, 512), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf529, buf532, 6656, 121, grid=grid(6656), stream=stream0)
        buf533 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf532, buf533, 512, 13, grid=grid(512), stream=stream0)
        buf534 = reinterpret_tensor(buf530, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf530  # reuse
        # Source Nodes: [x_244], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf534, addmm_32, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_32
        buf535 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf534, (1568, 2048), (2048, 1), 0), permute_358, out=buf535)
        del permute_358
        buf536 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf534, (2048, 1568), (1, 2048), 0), view_80, out=buf536)
        del view_80
        buf537 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf534, buf537, 26624, 121, grid=grid(26624), stream=stream0)
        buf538 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf537, buf538, 2048, 13, grid=grid(2048), stream=stream0)
        buf539 = buf513; del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf535, primals_55, buf539, 1568, 512, grid=grid(1568), stream=stream0)
        buf540 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf535, primals_55, mul_102, buf540, 6272, 128, grid=grid(6272), stream=stream0)
        buf541 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf540, buf541, 1568, 4, grid=grid(1568), stream=stream0)
        buf542 = reinterpret_tensor(buf532, (512, 13), (1, 512), 0); del buf532  # reuse
        buf544 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf535, mul_102, buf542, buf544, 6656, 121, grid=grid(6656), stream=stream0)
        buf543 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf542, buf543, 512, 13, grid=grid(512), stream=stream0)
        buf545 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf544, buf545, 512, 13, grid=grid(512), stream=stream0)
        buf546 = reinterpret_tensor(buf535, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf535  # reuse
        buf547 = reinterpret_tensor(buf528, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf546, div_22, primals_55, buf539, mul_102, buf541, buf547, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_22
        del mul_102
        del primals_55
        buf548 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf547, buf548, 6656, 121, grid=grid(6656), stream=stream0)
        buf549 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf548, buf549, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf550 = aten.convolution_backward(buf547, add_69, primals_221, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_69
        del primals_221
        buf551 = buf550[0]
        buf552 = buf550[1]
        del buf550
        buf554 = reinterpret_tensor(buf547, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_53.run(buf525, buf551, primals_54, buf554, 802816, grid=grid(802816), stream=stream0)
        del primals_54
        buf555 = reinterpret_tensor(buf546, (1568, 512), (512, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf554, buf555, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf556 = reinterpret_tensor(buf534, (1568, 2048), (2048, 1), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf555, permute_364, out=buf556)
        del permute_364
        buf560 = reinterpret_tensor(buf556, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf556  # reuse
        # Source Nodes: [x_230], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf560, addmm_30, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_30
        buf561 = reinterpret_tensor(buf554, (1568, 512), (512, 1), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (1568, 2048), (2048, 1), 0), permute_368, out=buf561)
        del permute_368
        buf565 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf561, primals_52, buf565, 1568, 512, grid=grid(1568), stream=stream0)
        buf566 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf561, primals_52, mul_96, buf566, 6272, 128, grid=grid(6272), stream=stream0)
        buf567 = buf539; del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf566, buf567, 1568, 4, grid=grid(1568), stream=stream0)
        buf573 = reinterpret_tensor(buf497, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_23, buf561, primals_52, buf565, mul_96, buf567, buf573, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_23
        del primals_52
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf576 = aten.convolution_backward(buf573, add_65, primals_215, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_65
        del primals_215
        buf577 = buf576[0]
        buf580 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_54.run(buf525, buf551, buf577, primals_51, buf580, 802816, grid=grid(802816), stream=stream0)
        del primals_51
        buf581 = reinterpret_tensor(buf445, (1568, 512), (512, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf580, buf581, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf582 = reinterpret_tensor(buf454, (1568, 2048), (2048, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf581, permute_374, out=buf582)
        del permute_374
        buf586 = reinterpret_tensor(buf582, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf582  # reuse
        # Source Nodes: [x_216], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf586, addmm_28, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_28
        buf587 = reinterpret_tensor(buf580, (1568, 512), (512, 1), 0); del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (1568, 2048), (2048, 1), 0), permute_378, out=buf587)
        del permute_378
        buf591 = buf567; del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf587, primals_49, buf591, 1568, 512, grid=grid(1568), stream=stream0)
        buf592 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf587, primals_49, mul_90, buf592, 6272, 128, grid=grid(6272), stream=stream0)
        buf593 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf592, buf593, 1568, 4, grid=grid(1568), stream=stream0)
        buf599 = reinterpret_tensor(buf518, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_24, buf587, primals_49, buf591, mul_90, buf593, buf599, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_24
        del primals_49
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf602 = aten.convolution_backward(buf599, add_61, primals_209, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_61
        del primals_209
        buf603 = buf602[0]
        buf553 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf579 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf605 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_55.run(buf525, buf551, addmm_31, buf577, addmm_29, buf603, addmm_27, buf553, buf579, buf605, 512, 1568, grid=grid(512), stream=stream0)
        del addmm_27
        del addmm_29
        del addmm_31
        buf557 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (512, 1568), (1, 512), 0), view_77, out=buf557)
        del view_77
        buf558 = reinterpret_tensor(buf548, (1, 512, 13), (6656, 1, 512), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf555, buf558, 6656, 121, grid=grid(6656), stream=stream0)
        del buf555
        buf559 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf558, buf559, 512, 13, grid=grid(512), stream=stream0)
        buf562 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (2048, 1568), (1, 2048), 0), view_75, out=buf562)
        del view_75
        buf563 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf560, buf563, 26624, 121, grid=grid(26624), stream=stream0)
        buf564 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf563, buf564, 2048, 13, grid=grid(2048), stream=stream0)
        buf568 = reinterpret_tensor(buf558, (512, 13), (1, 512), 0); del buf558  # reuse
        buf570 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf561, mul_96, buf568, buf570, 6656, 121, grid=grid(6656), stream=stream0)
        del buf561
        del mul_96
        buf569 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf568, buf569, 512, 13, grid=grid(512), stream=stream0)
        buf571 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf570, buf571, 512, 13, grid=grid(512), stream=stream0)
        buf574 = buf570; del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf573, buf574, 6656, 121, grid=grid(6656), stream=stream0)
        del buf573
        buf575 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf574, buf575, 512, 13, grid=grid(512), stream=stream0)
        buf578 = buf576[1]
        del buf576
        buf583 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (512, 1568), (1, 512), 0), view_72, out=buf583)
        del view_72
        buf584 = reinterpret_tensor(buf574, (1, 512, 13), (6656, 1, 512), 0); del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf581, buf584, 6656, 121, grid=grid(6656), stream=stream0)
        del buf581
        buf585 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf584, buf585, 512, 13, grid=grid(512), stream=stream0)
        buf588 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (2048, 1568), (1, 2048), 0), view_70, out=buf588)
        del view_70
        buf589 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf586, buf589, 26624, 121, grid=grid(26624), stream=stream0)
        buf590 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf589, buf590, 2048, 13, grid=grid(2048), stream=stream0)
        buf594 = reinterpret_tensor(buf584, (512, 13), (1, 512), 0); del buf584  # reuse
        buf596 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf587, mul_90, buf594, buf596, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_90
        buf595 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf594, buf595, 512, 13, grid=grid(512), stream=stream0)
        buf597 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf596, buf597, 512, 13, grid=grid(512), stream=stream0)
        buf600 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf599, buf600, 6656, 121, grid=grid(6656), stream=stream0)
        buf601 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf600, buf601, 512, 13, grid=grid(512), stream=stream0)
        buf604 = buf602[1]
        del buf602
        buf606 = reinterpret_tensor(buf599, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_56.run(buf525, buf551, buf577, buf603, primals_48, buf606, 802816, grid=grid(802816), stream=stream0)
        del primals_48
        buf607 = buf587; del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf606, buf607, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf608 = reinterpret_tensor(buf586, (1568, 2048), (2048, 1), 0); del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf607, permute_384, out=buf608)
        del permute_384
        buf609 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf607, (512, 1568), (1, 512), 0), view_67, out=buf609)
        del view_67
        buf610 = reinterpret_tensor(buf600, (1, 512, 13), (6656, 1, 512), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf607, buf610, 6656, 121, grid=grid(6656), stream=stream0)
        buf611 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf610, buf611, 512, 13, grid=grid(512), stream=stream0)
        buf612 = reinterpret_tensor(buf608, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf608  # reuse
        # Source Nodes: [x_202], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf612, addmm_26, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_26
        buf613 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf612, (1568, 2048), (2048, 1), 0), permute_388, out=buf613)
        del permute_388
        buf614 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf612, (2048, 1568), (1, 2048), 0), view_65, out=buf614)
        del view_65
        buf615 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf612, buf615, 26624, 121, grid=grid(26624), stream=stream0)
        buf616 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf615, buf616, 2048, 13, grid=grid(2048), stream=stream0)
        buf617 = buf593; del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf613, primals_46, buf617, 1568, 512, grid=grid(1568), stream=stream0)
        buf618 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf613, primals_46, mul_84, buf618, 6272, 128, grid=grid(6272), stream=stream0)
        buf619 = buf591; del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf618, buf619, 1568, 4, grid=grid(1568), stream=stream0)
        buf620 = reinterpret_tensor(buf610, (512, 13), (1, 512), 0); del buf610  # reuse
        buf622 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf613, mul_84, buf620, buf622, 6656, 121, grid=grid(6656), stream=stream0)
        buf621 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf620, buf621, 512, 13, grid=grid(512), stream=stream0)
        buf623 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf622, buf623, 512, 13, grid=grid(512), stream=stream0)
        buf624 = reinterpret_tensor(buf613, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf613  # reuse
        buf625 = reinterpret_tensor(buf606, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf624, div_25, primals_46, buf617, mul_84, buf619, buf625, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_25
        del mul_84
        del primals_46
        buf626 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf625, buf626, 6656, 121, grid=grid(6656), stream=stream0)
        buf627 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf626, buf627, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf628 = aten.convolution_backward(buf625, add_57, primals_203, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_57
        del primals_203
        buf629 = buf628[0]
        buf630 = buf628[1]
        del buf628
        buf631 = buf525; del buf525  # reuse
        buf634 = reinterpret_tensor(buf625, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_57.run(buf631, buf551, buf577, buf603, buf629, primals_45, buf634, 802816, grid=grid(802816), stream=stream0)
        del primals_45
        buf632 = reinterpret_tensor(buf626, (1, 512, 1, 1, 13), (6656, 13, 6656, 6656, 1), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf631, addmm_25, buf632, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_25
        buf633 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_47.run(buf632, buf633, 512, 13, grid=grid(512), stream=stream0)
        buf635 = reinterpret_tensor(buf629, (1568, 512), (512, 1), 0); del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf634, buf635, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf636 = reinterpret_tensor(buf612, (1568, 2048), (2048, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf635, permute_394, out=buf636)
        del permute_394
        buf637 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (512, 1568), (1, 512), 0), view_62, out=buf637)
        del view_62
        buf638 = reinterpret_tensor(buf632, (1, 512, 13), (6656, 1, 512), 0); del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf635, buf638, 6656, 121, grid=grid(6656), stream=stream0)
        buf639 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf638, buf639, 512, 13, grid=grid(512), stream=stream0)
        buf640 = reinterpret_tensor(buf636, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf636  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf640, addmm_24, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_24
        buf641 = buf635; del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (1568, 2048), (2048, 1), 0), permute_398, out=buf641)
        del permute_398
        buf642 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (2048, 1568), (1, 2048), 0), view_60, out=buf642)
        del view_60
        buf643 = buf615; del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf640, buf643, 26624, 121, grid=grid(26624), stream=stream0)
        buf644 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf643, buf644, 2048, 13, grid=grid(2048), stream=stream0)
        buf645 = buf619; del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf641, primals_43, buf645, 1568, 512, grid=grid(1568), stream=stream0)
        buf646 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf641, primals_43, mul_78, buf646, 6272, 128, grid=grid(6272), stream=stream0)
        buf647 = buf617; del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf646, buf647, 1568, 4, grid=grid(1568), stream=stream0)
        buf648 = reinterpret_tensor(buf638, (512, 13), (1, 512), 0); del buf638  # reuse
        buf650 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf641, mul_78, buf648, buf650, 6656, 121, grid=grid(6656), stream=stream0)
        buf649 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf648, buf649, 512, 13, grid=grid(512), stream=stream0)
        buf651 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf650, buf651, 512, 13, grid=grid(512), stream=stream0)
        buf652 = reinterpret_tensor(buf641, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf641  # reuse
        buf653 = reinterpret_tensor(buf634, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf652, div_26, primals_43, buf645, mul_78, buf647, buf653, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_26
        del mul_78
        del primals_43
        buf654 = buf650; del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf653, buf654, 6656, 121, grid=grid(6656), stream=stream0)
        buf655 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf654, buf655, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf656 = aten.convolution_backward(buf653, add_53, primals_197, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_53
        del primals_197
        buf657 = buf656[0]
        buf658 = buf656[1]
        del buf656
        buf660 = reinterpret_tensor(buf653, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_53.run(buf631, buf657, primals_42, buf660, 802816, grid=grid(802816), stream=stream0)
        del primals_42
        buf661 = reinterpret_tensor(buf652, (1568, 512), (512, 1), 0); del buf652  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf660, buf661, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf662 = reinterpret_tensor(buf640, (1568, 2048), (2048, 1), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf661, permute_404, out=buf662)
        del permute_404
        buf666 = reinterpret_tensor(buf662, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf662  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf666, addmm_22, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_22
        buf667 = reinterpret_tensor(buf660, (1568, 512), (512, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf666, (1568, 2048), (2048, 1), 0), permute_408, out=buf667)
        del permute_408
        buf671 = buf647; del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf667, primals_40, buf671, 1568, 512, grid=grid(1568), stream=stream0)
        buf672 = buf646; del buf646  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf667, primals_40, mul_72, buf672, 6272, 128, grid=grid(6272), stream=stream0)
        buf673 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf672, buf673, 1568, 4, grid=grid(1568), stream=stream0)
        buf679 = reinterpret_tensor(buf603, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_27, buf667, primals_40, buf671, mul_72, buf673, buf679, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_27
        del primals_40
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf682 = aten.convolution_backward(buf679, add_49, primals_191, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_49
        del primals_191
        buf683 = buf682[0]
        buf686 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_54.run(buf631, buf657, buf683, primals_39, buf686, 802816, grid=grid(802816), stream=stream0)
        del primals_39
        buf687 = reinterpret_tensor(buf551, (1568, 512), (512, 1), 0); del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf686, buf687, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf688 = reinterpret_tensor(buf560, (1568, 2048), (2048, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf687, permute_414, out=buf688)
        del permute_414
        buf692 = reinterpret_tensor(buf688, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf688  # reuse
        # Source Nodes: [x_160], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf692, addmm_20, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_20
        buf693 = reinterpret_tensor(buf686, (1568, 512), (512, 1), 0); del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf692, (1568, 2048), (2048, 1), 0), permute_418, out=buf693)
        del permute_418
        buf697 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf693, primals_37, buf697, 1568, 512, grid=grid(1568), stream=stream0)
        buf698 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf693, primals_37, mul_66, buf698, 6272, 128, grid=grid(6272), stream=stream0)
        buf699 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf698, buf699, 1568, 4, grid=grid(1568), stream=stream0)
        buf705 = reinterpret_tensor(buf624, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_28, buf693, primals_37, buf697, mul_66, buf699, buf705, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_28
        del primals_37
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf708 = aten.convolution_backward(buf705, add_45, primals_185, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_45
        del primals_185
        buf709 = buf708[0]
        buf659 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf685 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf711 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_55.run(buf631, buf657, addmm_23, buf683, addmm_21, buf709, addmm_19, buf659, buf685, buf711, 512, 1568, grid=grid(512), stream=stream0)
        del addmm_19
        del addmm_21
        del addmm_23
        buf663 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf661, (512, 1568), (1, 512), 0), view_57, out=buf663)
        del view_57
        buf664 = reinterpret_tensor(buf654, (1, 512, 13), (6656, 1, 512), 0); del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf661, buf664, 6656, 121, grid=grid(6656), stream=stream0)
        del buf661
        buf665 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf664, buf665, 512, 13, grid=grid(512), stream=stream0)
        buf668 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf666, (2048, 1568), (1, 2048), 0), view_55, out=buf668)
        del view_55
        buf669 = buf643; del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf666, buf669, 26624, 121, grid=grid(26624), stream=stream0)
        buf670 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf669, buf670, 2048, 13, grid=grid(2048), stream=stream0)
        buf674 = reinterpret_tensor(buf664, (512, 13), (1, 512), 0); del buf664  # reuse
        buf676 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf667, mul_72, buf674, buf676, 6656, 121, grid=grid(6656), stream=stream0)
        del buf667
        del mul_72
        buf675 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf674, buf675, 512, 13, grid=grid(512), stream=stream0)
        buf677 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf676, buf677, 512, 13, grid=grid(512), stream=stream0)
        buf680 = buf676; del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf679, buf680, 6656, 121, grid=grid(6656), stream=stream0)
        del buf679
        buf681 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf680, buf681, 512, 13, grid=grid(512), stream=stream0)
        buf684 = buf682[1]
        del buf682
        buf689 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf687, (512, 1568), (1, 512), 0), view_52, out=buf689)
        del view_52
        buf690 = reinterpret_tensor(buf680, (1, 512, 13), (6656, 1, 512), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf687, buf690, 6656, 121, grid=grid(6656), stream=stream0)
        del buf687
        buf691 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf690, buf691, 512, 13, grid=grid(512), stream=stream0)
        buf694 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf692, (2048, 1568), (1, 2048), 0), view_50, out=buf694)
        del view_50
        buf695 = buf669; del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf692, buf695, 26624, 121, grid=grid(26624), stream=stream0)
        buf696 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf695, buf696, 2048, 13, grid=grid(2048), stream=stream0)
        buf700 = reinterpret_tensor(buf690, (512, 13), (1, 512), 0); del buf690  # reuse
        buf702 = buf674; del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf693, mul_66, buf700, buf702, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_66
        buf701 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf700, buf701, 512, 13, grid=grid(512), stream=stream0)
        buf703 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf702, buf703, 512, 13, grid=grid(512), stream=stream0)
        buf706 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf705, buf706, 6656, 121, grid=grid(6656), stream=stream0)
        buf707 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf706, buf707, 512, 13, grid=grid(512), stream=stream0)
        buf710 = buf708[1]
        del buf708
        buf712 = reinterpret_tensor(buf705, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_56.run(buf631, buf657, buf683, buf709, primals_36, buf712, 802816, grid=grid(802816), stream=stream0)
        del primals_36
        buf713 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf712, buf713, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf714 = reinterpret_tensor(buf692, (1568, 2048), (2048, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf713, permute_424, out=buf714)
        del permute_424
        buf715 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf713, (512, 1568), (1, 512), 0), view_47, out=buf715)
        del view_47
        buf716 = reinterpret_tensor(buf706, (1, 512, 13), (6656, 1, 512), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf713, buf716, 6656, 121, grid=grid(6656), stream=stream0)
        buf717 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf716, buf717, 512, 13, grid=grid(512), stream=stream0)
        buf718 = reinterpret_tensor(buf714, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf714  # reuse
        # Source Nodes: [x_146], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf718, addmm_18, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_18
        buf719 = buf713; del buf713  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf718, (1568, 2048), (2048, 1), 0), permute_428, out=buf719)
        del permute_428
        buf720 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf718, (2048, 1568), (1, 2048), 0), view_45, out=buf720)
        del view_45
        buf721 = buf695; del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf718, buf721, 26624, 121, grid=grid(26624), stream=stream0)
        buf722 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf721, buf722, 2048, 13, grid=grid(2048), stream=stream0)
        buf723 = buf699; del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf719, primals_34, buf723, 1568, 512, grid=grid(1568), stream=stream0)
        buf724 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf719, primals_34, mul_60, buf724, 6272, 128, grid=grid(6272), stream=stream0)
        buf725 = buf697; del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf724, buf725, 1568, 4, grid=grid(1568), stream=stream0)
        buf726 = reinterpret_tensor(buf716, (512, 13), (1, 512), 0); del buf716  # reuse
        buf728 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf719, mul_60, buf726, buf728, 6656, 121, grid=grid(6656), stream=stream0)
        buf727 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf726, buf727, 512, 13, grid=grid(512), stream=stream0)
        buf729 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf728, buf729, 512, 13, grid=grid(512), stream=stream0)
        buf730 = reinterpret_tensor(buf719, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf719  # reuse
        buf731 = reinterpret_tensor(buf712, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf730, div_29, primals_34, buf723, mul_60, buf725, buf731, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del buf730
        del div_29
        del mul_60
        del primals_34
        buf732 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf731, buf732, 6656, 121, grid=grid(6656), stream=stream0)
        buf733 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf732, buf733, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf734 = aten.convolution_backward(buf731, add_41, primals_179, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_41
        del primals_179
        buf735 = buf734[0]
        buf736 = buf734[1]
        del buf734
        buf737 = buf631; del buf631  # reuse
        buf740 = reinterpret_tensor(buf731, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_57.run(buf737, buf657, buf683, buf709, buf735, primals_33, buf740, 802816, grid=grid(802816), stream=stream0)
        del buf657
        del buf683
        del primals_33
        buf738 = reinterpret_tensor(buf732, (1, 512, 1, 1, 13), (6656, 13, 6656, 6656, 1), 0); del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf737, addmm_17, buf738, 6656, 121, grid=grid(6656), stream=stream0)
        del addmm_17
        buf739 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_47.run(buf738, buf739, 512, 13, grid=grid(512), stream=stream0)
        buf741 = reinterpret_tensor(buf735, (1568, 512), (512, 1), 0); del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf740, buf741, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf742 = reinterpret_tensor(buf718, (1568, 2048), (2048, 1), 0); del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf741, permute_434, out=buf742)
        del permute_434
        buf743 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (512, 1568), (1, 512), 0), view_42, out=buf743)
        del view_42
        buf744 = reinterpret_tensor(buf738, (1, 512, 13), (6656, 1, 512), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf741, buf744, 6656, 121, grid=grid(6656), stream=stream0)
        buf745 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf744, buf745, 512, 13, grid=grid(512), stream=stream0)
        buf746 = reinterpret_tensor(buf742, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf742  # reuse
        # Source Nodes: [x_132], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf746, addmm_16, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_16
        buf747 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (1568, 2048), (2048, 1), 0), permute_438, out=buf747)
        del permute_438
        buf748 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (2048, 1568), (1, 2048), 0), view_40, out=buf748)
        del view_40
        buf749 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf746, buf749, 26624, 121, grid=grid(26624), stream=stream0)
        buf750 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf749, buf750, 2048, 13, grid=grid(2048), stream=stream0)
        buf751 = buf725; del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf747, primals_31, buf751, 1568, 512, grid=grid(1568), stream=stream0)
        buf752 = buf724; del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf747, primals_31, mul_54, buf752, 6272, 128, grid=grid(6272), stream=stream0)
        buf753 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf752, buf753, 1568, 4, grid=grid(1568), stream=stream0)
        buf754 = reinterpret_tensor(buf744, (512, 13), (1, 512), 0); del buf744  # reuse
        buf756 = buf726; del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf747, mul_54, buf754, buf756, 6656, 121, grid=grid(6656), stream=stream0)
        buf755 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf754, buf755, 512, 13, grid=grid(512), stream=stream0)
        buf757 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf756, buf757, 512, 13, grid=grid(512), stream=stream0)
        buf758 = reinterpret_tensor(buf747, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf747  # reuse
        buf759 = reinterpret_tensor(buf740, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf758, div_30, primals_31, buf751, mul_54, buf753, buf759, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_30
        del mul_54
        del primals_31
        buf760 = buf756; del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf759, buf760, 6656, 121, grid=grid(6656), stream=stream0)
        buf761 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf760, buf761, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf762 = aten.convolution_backward(buf759, add_37, primals_173, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_37
        del primals_173
        buf763 = buf762[0]
        buf764 = buf762[1]
        del buf762
        buf766 = reinterpret_tensor(buf759, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_53.run(buf737, buf763, primals_30, buf766, 802816, grid=grid(802816), stream=stream0)
        del primals_30
        buf767 = reinterpret_tensor(buf758, (1568, 512), (512, 1), 0); del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf766, buf767, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf768 = reinterpret_tensor(buf746, (1568, 2048), (2048, 1), 0); del buf746  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf767, permute_444, out=buf768)
        del permute_444
        buf772 = reinterpret_tensor(buf768, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf768  # reuse
        # Source Nodes: [x_118], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf772, addmm_14, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_14
        buf773 = reinterpret_tensor(buf766, (1568, 512), (512, 1), 0); del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (1568, 2048), (2048, 1), 0), permute_448, out=buf773)
        del permute_448
        buf777 = buf753; del buf753  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf773, primals_28, buf777, 1568, 512, grid=grid(1568), stream=stream0)
        buf778 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf773, primals_28, mul_48, buf778, 6272, 128, grid=grid(6272), stream=stream0)
        buf779 = buf751; del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf778, buf779, 1568, 4, grid=grid(1568), stream=stream0)
        buf785 = reinterpret_tensor(buf709, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_35.run(div_31, buf773, primals_28, buf777, mul_48, buf779, buf785, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del div_31
        del primals_28
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf788 = aten.convolution_backward(buf785, add_33, primals_167, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del add_33
        del primals_167
        buf789 = buf788[0]
        buf765 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf791 = empty((1, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_58.run(buf737, buf763, addmm_15, buf789, addmm_13, buf765, buf791, 512, 1568, grid=grid(512), stream=stream0)
        del addmm_13
        del addmm_15
        buf769 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf767, (512, 1568), (1, 512), 0), view_37, out=buf769)
        del view_37
        buf770 = reinterpret_tensor(buf760, (1, 512, 13), (6656, 1, 512), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf767, buf770, 6656, 121, grid=grid(6656), stream=stream0)
        del buf767
        buf771 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf770, buf771, 512, 13, grid=grid(512), stream=stream0)
        buf774 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (2048, 1568), (1, 2048), 0), view_35, out=buf774)
        del view_35
        buf775 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf772, buf775, 26624, 121, grid=grid(26624), stream=stream0)
        buf776 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf775, buf776, 2048, 13, grid=grid(2048), stream=stream0)
        buf780 = reinterpret_tensor(buf770, (512, 13), (1, 512), 0); del buf770  # reuse
        buf782 = buf754; del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf773, mul_48, buf780, buf782, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_48
        buf781 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf780, buf781, 512, 13, grid=grid(512), stream=stream0)
        buf783 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf782, buf783, 512, 13, grid=grid(512), stream=stream0)
        buf786 = buf782; del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf785, buf786, 6656, 121, grid=grid(6656), stream=stream0)
        buf787 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf786, buf787, 512, 13, grid=grid(512), stream=stream0)
        buf790 = buf788[1]
        del buf788
        buf792 = reinterpret_tensor(buf785, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf785  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_54.run(buf737, buf763, buf789, primals_27, buf792, 802816, grid=grid(802816), stream=stream0)
        del primals_27
        buf793 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf792, buf793, 1568, 512, grid=grid(1568, 512), stream=stream0)
        buf794 = reinterpret_tensor(buf772, (1568, 2048), (2048, 1), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf793, permute_454, out=buf794)
        del permute_454
        buf795 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf793, (512, 1568), (1, 512), 0), view_32, out=buf795)
        del view_32
        buf796 = reinterpret_tensor(buf786, (1, 512, 13), (6656, 1, 512), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_38.run(buf793, buf796, 6656, 121, grid=grid(6656), stream=stream0)
        buf797 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_37.run(buf796, buf797, 512, 13, grid=grid(512), stream=stream0)
        buf798 = reinterpret_tensor(buf794, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf794  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf798, addmm_12, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_12
        buf799 = buf793; del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (1568, 2048), (2048, 1), 0), permute_458, out=buf799)
        del permute_458
        buf800 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (2048, 1568), (1, 2048), 0), view_30, out=buf800)
        del view_30
        buf801 = buf775; del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf798, buf801, 26624, 121, grid=grid(26624), stream=stream0)
        buf802 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf801, buf802, 2048, 13, grid=grid(2048), stream=stream0)
        del buf801
        buf803 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_33.run(buf799, primals_25, buf803, 1568, 512, grid=grid(1568), stream=stream0)
        buf804 = buf778; del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_34.run(buf799, primals_25, mul_42, buf804, 6272, 128, grid=grid(6272), stream=stream0)
        buf805 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_27.run(buf804, buf805, 1568, 4, grid=grid(1568), stream=stream0)
        buf806 = reinterpret_tensor(buf796, (512, 13), (1, 512), 0); del buf796  # reuse
        buf808 = buf780; del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf799, mul_42, buf806, buf808, 6656, 121, grid=grid(6656), stream=stream0)
        buf807 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf806, buf807, 512, 13, grid=grid(512), stream=stream0)
        del buf806
        buf809 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_37.run(buf808, buf809, 512, 13, grid=grid(512), stream=stream0)
        buf810 = reinterpret_tensor(buf799, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf799  # reuse
        buf811 = reinterpret_tensor(buf792, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_44.run(buf810, div_32, primals_25, buf803, mul_42, buf805, buf811, 112, 7168, grid=grid(112, 7168), stream=stream0)
        del buf803
        del buf805
        del buf810
        del div_32
        del mul_42
        del primals_25
        buf812 = buf808; del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf811, buf812, 6656, 121, grid=grid(6656), stream=stream0)
        buf813 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf812, buf813, 512, 13, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf814 = aten.convolution_backward(buf811, convolution_8, primals_161, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False])
        del convolution_8
        del primals_161
        buf815 = buf814[0]
        buf816 = buf814[1]
        del buf814
        buf817 = buf811; del buf811  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf737, buf763, buf789, buf815, buf817, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf737
        del buf763
        del buf789
        del buf815
        buf818 = buf812; del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_45.run(buf817, buf818, 6656, 121, grid=grid(6656), stream=stream0)
        buf819 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_37.run(buf818, buf819, 512, 13, grid=grid(512), stream=stream0)
        del buf818
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf820 = aten.convolution_backward(buf817, permute_29, primals_159, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf817
        del permute_29
        del primals_159
        buf821 = buf820[0]
        buf822 = buf820[1]
        del buf820
        buf823 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_60.run(buf821, primals_23, buf823, 12544, 128, grid=grid(12544), stream=stream0)
        buf824 = reinterpret_tensor(buf804, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_61.run(buf823, buf824, 6272, 2, grid=grid(6272), stream=stream0)
        buf825 = reinterpret_tensor(buf823, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_62.run(buf821, primals_23, mul_40, buf825, 12544, 128, grid=grid(12544), stream=stream0)
        buf826 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_63.run(buf825, buf826, 6272, 2, grid=grid(6272), stream=stream0)
        buf827 = empty((256, ), device='cuda', dtype=torch.float32)
        buf828 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_64.run(buf821, mul_40, buf827, buf828, 256, 6272, grid=grid(256), stream=stream0)
        buf829 = reinterpret_tensor(buf69, (8, 28, 28, 256), (200704, 7168, 256, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_backward_65.run(div_33, buf821, primals_23, buf824, mul_40, buf826, buf829, 224, 7168, grid=grid(224, 7168), stream=stream0)
        del div_33
        del mul_40
        del primals_23
        buf832 = buf821; del buf821  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_66.run(buf829, primals_22, buf832, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del primals_22
        buf833 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_67.run(buf832, buf833, 6272, 256, grid=grid(6272, 256), stream=stream0)
        buf834 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf833, permute_466, out=buf834)
        del permute_466
        buf838 = reinterpret_tensor(buf834, (8, 28, 28, 1024), (802816, 28672, 1024, 1), 0); del buf834  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_68.run(buf838, addmm_10, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_10
        buf839 = reinterpret_tensor(buf832, (6272, 256), (256, 1), 0); del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf838, (6272, 1024), (1024, 1), 0), permute_470, out=buf839)
        del permute_470
        buf843 = buf826; del buf826  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_69.run(buf839, primals_20, buf843, 6272, 256, grid=grid(6272), stream=stream0)
        buf844 = buf825; del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_70.run(buf839, primals_20, mul_34, buf844, 12544, 128, grid=grid(12544), stream=stream0)
        buf845 = buf824; del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_63.run(buf844, buf845, 6272, 2, grid=grid(6272), stream=stream0)
        buf851 = empty_strided((8, 256, 28, 28), (200704, 1, 256, 7168), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_71.run(div_34, buf839, primals_20, buf843, mul_34, buf845, buf851, 224, 7168, grid=grid(224, 7168), stream=stream0)
        del div_34
        del primals_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf854 = aten.convolution_backward(buf851, add_23, primals_153, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, False])
        del add_23
        del primals_153
        buf855 = buf854[0]
        buf830 = reinterpret_tensor(buf844, (1, 256, 1, 1, 49), (12544, 1, 12544, 12544, 256), 0); del buf844  # reuse
        buf857 = empty_strided((1, 256, 1, 1, 49), (12544, 1, 12544, 12544, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_72.run(buf829, addmm_11, buf855, addmm_9, buf830, buf857, 12544, 128, grid=grid(12544), stream=stream0)
        del addmm_11
        del addmm_9
        buf831 = empty((1, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_73.run(buf830, buf831, 256, 49, grid=grid(256), stream=stream0)
        buf835 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf833, (256, 6272), (1, 256), 0), view_27, out=buf835)
        del view_27
        buf836 = reinterpret_tensor(buf830, (1, 256, 49), (12544, 1, 256), 0); del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_74.run(buf833, buf836, 12544, 128, grid=grid(12544), stream=stream0)
        del buf833
        buf837 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_73.run(buf836, buf837, 256, 49, grid=grid(256), stream=stream0)
        buf840 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf838, (1024, 6272), (1, 1024), 0), view_25, out=buf840)
        del view_25
        buf841 = empty_strided((1, 1024, 49), (50176, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_75.run(buf838, buf841, 50176, 128, grid=grid(50176), stream=stream0)
        buf842 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_76.run(buf841, buf842, 1024, 49, grid=grid(1024), stream=stream0)
        buf846 = reinterpret_tensor(buf836, (256, 49), (1, 256), 0); del buf836  # reuse
        buf848 = empty_strided((256, 49), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_77.run(buf839, mul_34, buf846, buf848, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_34
        buf847 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_73.run(buf846, buf847, 256, 49, grid=grid(256), stream=stream0)
        del buf846
        buf849 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_73.run(buf848, buf849, 256, 49, grid=grid(256), stream=stream0)
        buf852 = buf848; del buf848  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_78.run(buf851, buf852, 12544, 128, grid=grid(12544), stream=stream0)
        buf853 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_73.run(buf852, buf853, 256, 49, grid=grid(256), stream=stream0)
        buf856 = buf854[1]
        del buf854
        buf858 = empty((1, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_73.run(buf857, buf858, 256, 49, grid=grid(256), stream=stream0)
        buf859 = reinterpret_tensor(buf851, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_79.run(buf829, buf855, primals_19, buf859, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del primals_19
        buf860 = buf839; del buf839  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_67.run(buf859, buf860, 6272, 256, grid=grid(6272, 256), stream=stream0)
        buf861 = reinterpret_tensor(buf838, (6272, 1024), (1024, 1), 0); del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf860, permute_476, out=buf861)
        del permute_476
        buf862 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf860, (256, 6272), (1, 256), 0), view_22, out=buf862)
        del view_22
        buf863 = reinterpret_tensor(buf857, (1, 256, 49), (12544, 1, 256), 0); del buf857  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_74.run(buf860, buf863, 12544, 128, grid=grid(12544), stream=stream0)
        buf864 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_73.run(buf863, buf864, 256, 49, grid=grid(256), stream=stream0)
        buf865 = reinterpret_tensor(buf861, (8, 28, 28, 1024), (802816, 28672, 1024, 1), 0); del buf861  # reuse
        # Source Nodes: [x_71], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_68.run(buf865, addmm_8, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_8
        buf866 = buf860; del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (6272, 1024), (1024, 1), 0), permute_480, out=buf866)
        del permute_480
        buf867 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (1024, 6272), (1, 1024), 0), view_20, out=buf867)
        del view_20
        buf868 = buf841; del buf841  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_75.run(buf865, buf868, 50176, 128, grid=grid(50176), stream=stream0)
        buf869 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_76.run(buf868, buf869, 1024, 49, grid=grid(1024), stream=stream0)
        buf870 = buf845; del buf845  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_69.run(buf866, primals_17, buf870, 6272, 256, grid=grid(6272), stream=stream0)
        buf871 = reinterpret_tensor(buf863, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_70.run(buf866, primals_17, mul_28, buf871, 12544, 128, grid=grid(12544), stream=stream0)
        buf872 = buf843; del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_63.run(buf871, buf872, 6272, 2, grid=grid(6272), stream=stream0)
        buf873 = reinterpret_tensor(buf871, (256, 49), (1, 256), 0); del buf871  # reuse
        buf875 = buf852; del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_77.run(buf866, mul_28, buf873, buf875, 12544, 128, grid=grid(12544), stream=stream0)
        buf874 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_73.run(buf873, buf874, 256, 49, grid=grid(256), stream=stream0)
        buf876 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_73.run(buf875, buf876, 256, 49, grid=grid(256), stream=stream0)
        buf877 = reinterpret_tensor(buf866, (8, 28, 28, 256), (200704, 7168, 256, 1), 0); del buf866  # reuse
        buf878 = reinterpret_tensor(buf859, (8, 256, 28, 28), (200704, 1, 256, 7168), 0); del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_80.run(buf877, div_35, primals_17, buf870, mul_28, buf872, buf878, 224, 7168, grid=grid(224, 7168), stream=stream0)
        del div_35
        del mul_28
        del primals_17
        buf879 = buf875; del buf875  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_81.run(buf878, buf879, 12544, 128, grid=grid(12544), stream=stream0)
        buf880 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_73.run(buf879, buf880, 256, 49, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf881 = aten.convolution_backward(buf878, add_19, primals_147, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, False])
        del add_19
        del primals_147
        buf882 = buf881[0]
        buf883 = buf881[1]
        del buf881
        buf884 = reinterpret_tensor(buf879, (1, 256, 1, 1, 49), (12544, 49, 12544, 12544, 1), 0); del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_82.run(buf829, buf855, buf882, addmm_7, buf884, 12544, 128, grid=grid(12544), stream=stream0)
        del addmm_7
        buf885 = empty((1, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_83.run(buf884, buf885, 256, 49, grid=grid(256), stream=stream0)
        buf886 = reinterpret_tensor(buf878, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_84.run(buf829, buf855, buf882, primals_16, buf886, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del primals_16
        buf887 = reinterpret_tensor(buf877, (6272, 256), (256, 1), 0); del buf877  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_67.run(buf886, buf887, 6272, 256, grid=grid(6272, 256), stream=stream0)
        buf888 = reinterpret_tensor(buf865, (6272, 1024), (1024, 1), 0); del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf887, permute_486, out=buf888)
        del permute_486
        buf889 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf887, (256, 6272), (1, 256), 0), view_17, out=buf889)
        del view_17
        buf890 = reinterpret_tensor(buf884, (1, 256, 49), (12544, 1, 256), 0); del buf884  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_74.run(buf887, buf890, 12544, 128, grid=grid(12544), stream=stream0)
        buf891 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_73.run(buf890, buf891, 256, 49, grid=grid(256), stream=stream0)
        buf892 = reinterpret_tensor(buf888, (8, 28, 28, 1024), (802816, 28672, 1024, 1), 0); del buf888  # reuse
        # Source Nodes: [x_57], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_68.run(buf892, addmm_6, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_6
        buf893 = buf887; del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf892, (6272, 1024), (1024, 1), 0), permute_490, out=buf893)
        del permute_490
        buf894 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf892, (1024, 6272), (1, 1024), 0), view_15, out=buf894)
        del view_15
        buf895 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_75.run(buf892, buf895, 50176, 128, grid=grid(50176), stream=stream0)
        del buf892
        buf896 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_76.run(buf895, buf896, 1024, 49, grid=grid(1024), stream=stream0)
        del buf895
        buf897 = buf872; del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_69.run(buf893, primals_14, buf897, 6272, 256, grid=grid(6272), stream=stream0)
        buf898 = reinterpret_tensor(buf890, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf890  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_70.run(buf893, primals_14, mul_22, buf898, 12544, 128, grid=grid(12544), stream=stream0)
        buf899 = buf870; del buf870  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_63.run(buf898, buf899, 6272, 2, grid=grid(6272), stream=stream0)
        buf900 = reinterpret_tensor(buf898, (256, 49), (1, 256), 0); del buf898  # reuse
        buf902 = buf873; del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_77.run(buf893, mul_22, buf900, buf902, 12544, 128, grid=grid(12544), stream=stream0)
        buf901 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_73.run(buf900, buf901, 256, 49, grid=grid(256), stream=stream0)
        del buf900
        buf903 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_73.run(buf902, buf903, 256, 49, grid=grid(256), stream=stream0)
        buf904 = reinterpret_tensor(buf893, (8, 28, 28, 256), (200704, 7168, 256, 1), 0); del buf893  # reuse
        buf905 = reinterpret_tensor(buf886, (8, 256, 28, 28), (200704, 1, 256, 7168), 0); del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_80.run(buf904, div_36, primals_14, buf897, mul_22, buf899, buf905, 224, 7168, grid=grid(224, 7168), stream=stream0)
        del buf897
        del buf899
        del buf904
        del div_36
        del mul_22
        del primals_14
        buf906 = buf902; del buf902  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_81.run(buf905, buf906, 12544, 128, grid=grid(12544), stream=stream0)
        buf907 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_73.run(buf906, buf907, 256, 49, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf908 = aten.convolution_backward(buf905, convolution_4, primals_141, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, False])
        del convolution_4
        del primals_141
        buf909 = buf908[0]
        buf910 = buf908[1]
        del buf908
        buf911 = buf905; del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_85.run(buf829, buf855, buf882, buf909, buf911, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf829
        del buf855
        del buf882
        del buf909
        buf912 = buf906; del buf906  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_81.run(buf911, buf912, 12544, 128, grid=grid(12544), stream=stream0)
        buf913 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_73.run(buf912, buf913, 256, 49, grid=grid(256), stream=stream0)
        del buf912
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf914 = aten.convolution_backward(buf911, permute_15, primals_139, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf911
        del permute_15
        del primals_139
        buf915 = buf914[0]
        buf916 = buf914[1]
        del buf914
        buf924 = reinterpret_tensor(buf798, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_86.run(buf915, primals_12, mul_20, div_37, buf924, 25088, 128, grid=grid(25088), stream=stream0)
        del div_37
        del primals_12
        buf919 = empty_strided((128, 4, 49), (49, 6272, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_87.run(buf915, mul_20, buf919, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_20
        buf920 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_88.run(buf919, buf920, 512, 49, grid=grid(512), stream=stream0)
        buf921 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_89.run(buf920, buf921, 128, 4, grid=grid(128), stream=stream0)
        buf922 = buf920; del buf920  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_90.run(buf915, buf922, 512, 6272, grid=grid(512), stream=stream0)
        buf923 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_89.run(buf922, buf923, 128, 4, grid=grid(128), stream=stream0)
        buf927 = buf915; del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_91.run(buf924, primals_11, buf927, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_11
        buf928 = reinterpret_tensor(buf666, (25088, 128), (128, 1), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_92.run(buf927, buf928, 25088, 128, grid=grid(25088, 128), stream=stream0)
        buf929 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf928, permute_498, out=buf929)
        del permute_498
        buf933 = reinterpret_tensor(buf929, (8, 56, 56, 512), (1605632, 28672, 512, 1), 0); del buf929  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_93.run(buf933, addmm_4, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_4
        buf934 = reinterpret_tensor(buf927, (25088, 128), (128, 1), 0); del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf933, (25088, 512), (512, 1), 0), permute_502, out=buf934)
        del permute_502
        buf938 = reinterpret_tensor(buf919, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf919  # reuse
        buf939 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_94.run(buf934, primals_9, mul_14, buf938, buf939, 25088, 128, grid=grid(25088), stream=stream0)
        buf945 = empty_strided((8, 128, 56, 56), (401408, 1, 128, 7168), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_95.run(div_38, buf934, primals_9, buf938, mul_14, buf939, buf945, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del div_38
        del primals_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf948 = aten.convolution_backward(buf945, add_9, primals_133, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, False])
        del add_9
        del primals_133
        buf949 = buf948[0]
        buf925 = reinterpret_tensor(buf939, (1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), 0); del buf939  # reuse
        buf951 = reinterpret_tensor(buf938, (1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), 0); del buf938  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_96.run(buf924, addmm_5, buf949, addmm_3, buf925, buf951, 25088, 128, grid=grid(25088), stream=stream0)
        del addmm_3
        del addmm_5
        buf926 = empty((1, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_97.run(buf925, buf926, 128, 196, grid=grid(128), stream=stream0)
        buf930 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf928, (128, 25088), (1, 128), 0), view_12, out=buf930)
        del view_12
        buf931 = reinterpret_tensor(buf925, (1, 128, 196), (25088, 1, 128), 0); del buf925  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_98.run(buf928, buf931, 25088, 128, grid=grid(25088), stream=stream0)
        del buf928
        buf932 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_mul_sum_97.run(buf931, buf932, 128, 196, grid=grid(128), stream=stream0)
        buf935 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf933, (512, 25088), (1, 512), 0), view_10, out=buf935)
        del view_10
        buf936 = empty_strided((1, 512, 196), (100352, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_99.run(buf933, buf936, 100352, 128, grid=grid(100352), stream=stream0)
        buf937 = reinterpret_tensor(buf922, (1, 512), (512, 1), 0); del buf922  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_100.run(buf936, buf937, 512, 196, grid=grid(512), stream=stream0)
        buf940 = reinterpret_tensor(buf931, (128, 196), (1, 128), 0); del buf931  # reuse
        buf942 = empty_strided((128, 196), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_101.run(buf934, mul_14, buf940, buf942, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_14
        buf941 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_mul_sum_97.run(buf940, buf941, 128, 196, grid=grid(128), stream=stream0)
        buf943 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_mul_sum_97.run(buf942, buf943, 128, 196, grid=grid(128), stream=stream0)
        buf946 = buf942; del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_102.run(buf945, buf946, 25088, 128, grid=grid(25088), stream=stream0)
        buf947 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_mul_sum_97.run(buf946, buf947, 128, 196, grid=grid(128), stream=stream0)
        buf950 = buf948[1]
        del buf948
        buf952 = empty((1, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_97.run(buf951, buf952, 128, 196, grid=grid(128), stream=stream0)
        buf953 = reinterpret_tensor(buf945, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_103.run(buf924, buf949, primals_8, buf953, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_8
        buf954 = buf934; del buf934  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_92.run(buf953, buf954, 25088, 128, grid=grid(25088, 128), stream=stream0)
        buf955 = reinterpret_tensor(buf933, (25088, 512), (512, 1), 0); del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf954, permute_508, out=buf955)
        del permute_508
        buf956 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf954, (128, 25088), (1, 128), 0), view_7, out=buf956)
        del view_7
        buf957 = reinterpret_tensor(buf951, (1, 128, 196), (25088, 1, 128), 0); del buf951  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_98.run(buf954, buf957, 25088, 128, grid=grid(25088), stream=stream0)
        buf958 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_mul_sum_97.run(buf957, buf958, 128, 196, grid=grid(128), stream=stream0)
        buf959 = reinterpret_tensor(buf955, (8, 56, 56, 512), (1605632, 28672, 512, 1), 0); del buf955  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_93.run(buf959, addmm_2, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_2
        buf960 = buf954; del buf954  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf959, (25088, 512), (512, 1), 0), permute_512, out=buf960)
        del permute_512
        buf961 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf959, (512, 25088), (1, 512), 0), view_5, out=buf961)
        del view_5
        buf962 = buf936; del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_99.run(buf959, buf962, 100352, 128, grid=grid(100352), stream=stream0)
        buf963 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_100.run(buf962, buf963, 512, 196, grid=grid(512), stream=stream0)
        buf964 = reinterpret_tensor(buf957, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf957  # reuse
        buf965 = reinterpret_tensor(buf946, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf946  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_94.run(buf960, primals_6, mul_8, buf964, buf965, 25088, 128, grid=grid(25088), stream=stream0)
        buf966 = buf940; del buf940  # reuse
        buf968 = empty_strided((128, 196), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_101.run(buf960, mul_8, buf966, buf968, 25088, 128, grid=grid(25088), stream=stream0)
        buf967 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_mul_sum_97.run(buf966, buf967, 128, 196, grid=grid(128), stream=stream0)
        buf969 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_mul_sum_97.run(buf968, buf969, 128, 196, grid=grid(128), stream=stream0)
        buf970 = reinterpret_tensor(buf960, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf960  # reuse
        buf971 = reinterpret_tensor(buf953, (8, 128, 56, 56), (401408, 1, 128, 7168), 0); del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_104.run(buf970, div_39, primals_6, buf964, mul_8, buf965, buf971, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del div_39
        del mul_8
        del primals_6
        buf972 = reinterpret_tensor(buf965, (128, 196), (1, 128), 0); del buf965  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_105.run(buf971, buf972, 25088, 128, grid=grid(25088), stream=stream0)
        buf973 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_mul_sum_97.run(buf972, buf973, 128, 196, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf974 = aten.convolution_backward(buf971, add_5, primals_127, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, False])
        del add_5
        del primals_127
        buf975 = buf974[0]
        buf976 = buf974[1]
        del buf974
        buf977 = reinterpret_tensor(buf972, (1, 128, 1, 1, 196), (25088, 196, 25088, 25088, 1), 0); del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_106.run(buf924, buf949, buf975, addmm_1, buf977, 25088, 128, grid=grid(25088), stream=stream0)
        del addmm_1
        buf978 = empty((1, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_107.run(buf977, buf978, 128, 196, grid=grid(128), stream=stream0)
        buf979 = reinterpret_tensor(buf971, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_108.run(buf924, buf949, buf975, primals_5, buf979, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_5
        buf980 = reinterpret_tensor(buf970, (25088, 128), (128, 1), 0); del buf970  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_92.run(buf979, buf980, 25088, 128, grid=grid(25088, 128), stream=stream0)
        buf981 = reinterpret_tensor(buf959, (25088, 512), (512, 1), 0); del buf959  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf980, permute_518, out=buf981)
        del permute_518
        buf982 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (128, 25088), (1, 128), 0), view_2, out=buf982)
        del view_2
        buf983 = reinterpret_tensor(buf977, (1, 128, 196), (25088, 1, 128), 0); del buf977  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_98.run(buf980, buf983, 25088, 128, grid=grid(25088), stream=stream0)
        buf984 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_mul_sum_97.run(buf983, buf984, 128, 196, grid=grid(128), stream=stream0)
        buf985 = reinterpret_tensor(buf981, (8, 56, 56, 512), (1605632, 28672, 512, 1), 0); del buf981  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_93.run(buf985, addmm, 12845056, grid=grid(12845056), stream=stream0)
        del addmm
        buf986 = buf980; del buf980  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (25088, 512), (512, 1), 0), permute_522, out=buf986)
        del permute_522
        buf987 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (512, 25088), (1, 512), 0), view, out=buf987)
        del view
        buf988 = buf962; del buf962  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_99.run(buf985, buf988, 100352, 128, grid=grid(100352), stream=stream0)
        del buf985
        buf989 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_100.run(buf988, buf989, 512, 196, grid=grid(512), stream=stream0)
        del buf988
        buf990 = reinterpret_tensor(buf983, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf983  # reuse
        buf991 = buf964; del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_94.run(buf986, primals_3, mul_2, buf990, buf991, 25088, 128, grid=grid(25088), stream=stream0)
        buf992 = buf968; del buf968  # reuse
        buf994 = buf966; del buf966  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_101.run(buf986, mul_2, buf992, buf994, 25088, 128, grid=grid(25088), stream=stream0)
        buf993 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_mul_sum_97.run(buf992, buf993, 128, 196, grid=grid(128), stream=stream0)
        del buf992
        buf995 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_mul_sum_97.run(buf994, buf995, 128, 196, grid=grid(128), stream=stream0)
        del buf994
        buf996 = reinterpret_tensor(buf986, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf986  # reuse
        buf997 = reinterpret_tensor(buf979, (8, 128, 56, 56), (401408, 1, 128, 7168), 0); del buf979  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_native_layer_norm_backward_permute_104.run(buf996, div_40, primals_3, buf990, mul_2, buf991, buf997, 448, 7168, grid=grid(448, 7168), stream=stream0)
        del buf990
        del div_40
        del mul_2
        del primals_3
        buf998 = reinterpret_tensor(buf991, (128, 196), (1, 128), 0); del buf991  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_105.run(buf997, buf998, 25088, 128, grid=grid(25088), stream=stream0)
        buf999 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_mul_sum_97.run(buf998, buf999, 128, 196, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1000 = aten.convolution_backward(buf997, permute_1, primals_121, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, False])
        del permute_1
        del primals_121
        buf1001 = buf1000[0]
        buf1002 = buf1000[1]
        del buf1000
        buf1003 = reinterpret_tensor(buf997, (8, 56, 56, 128), (401408, 56, 1, 3136), 0); del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_backward_109.run(buf924, buf949, buf975, buf1001, primals_1, buf1003, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del primals_1
        buf1011 = reinterpret_tensor(buf996, (8, 128, 56, 56), (401408, 1, 128, 7168), 0); del buf996  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.permute]
        triton_red_fused_native_layer_norm_backward_permute_110.run(buf1003, mul, div_41, buf1011, 25088, 128, grid=grid(25088), stream=stream0)
        del buf1003
        del div_41
        buf1006 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf1008 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_111.run(buf924, buf949, buf975, buf1001, mul, buf1006, buf1008, 512, 6272, grid=grid(512), stream=stream0)
        del buf1001
        del buf924
        del buf949
        del buf975
        del mul
        buf1007 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_89.run(buf1006, buf1007, 128, 4, grid=grid(128), stream=stream0)
        del buf1006
        buf1009 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_89.run(buf1008, buf1009, 128, 4, grid=grid(128), stream=stream0)
        del buf1008
        buf1012 = buf998; del buf998  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_105.run(buf1011, buf1012, 25088, 128, grid=grid(25088), stream=stream0)
        buf1013 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_mul_sum_97.run(buf1012, buf1013, 128, 196, grid=grid(128), stream=stream0)
        del buf1012
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1014 = aten.convolution_backward(buf1011, primals_345, primals_119, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1011
        del primals_119
        del primals_345
        buf1015 = buf1014[1]
        return (buf1007, buf1009, buf993, buf995, reinterpret_tensor(buf978, (128, ), (1, ), 0), buf967, buf969, reinterpret_tensor(buf952, (128, ), (1, ), 0), buf941, buf943, reinterpret_tensor(buf926, (128, ), (1, ), 0), buf921, buf923, buf901, buf903, reinterpret_tensor(buf885, (256, ), (1, ), 0), buf874, buf876, reinterpret_tensor(buf858, (256, ), (1, ), 0), buf847, buf849, reinterpret_tensor(buf831, (256, ), (1, ), 0), buf827, buf828, buf807, buf809, reinterpret_tensor(buf791, (512, ), (1, ), 0), buf781, buf783, reinterpret_tensor(buf765, (512, ), (1, ), 0), buf755, buf757, reinterpret_tensor(buf739, (512, ), (1, ), 0), buf727, buf729, reinterpret_tensor(buf711, (512, ), (1, ), 0), buf701, buf703, reinterpret_tensor(buf685, (512, ), (1, ), 0), buf675, buf677, reinterpret_tensor(buf659, (512, ), (1, ), 0), buf649, buf651, reinterpret_tensor(buf633, (512, ), (1, ), 0), buf621, buf623, reinterpret_tensor(buf605, (512, ), (1, ), 0), buf595, buf597, reinterpret_tensor(buf579, (512, ), (1, ), 0), buf569, buf571, reinterpret_tensor(buf553, (512, ), (1, ), 0), buf543, buf545, reinterpret_tensor(buf527, (512, ), (1, ), 0), buf515, buf517, reinterpret_tensor(buf499, (512, ), (1, ), 0), buf489, buf491, reinterpret_tensor(buf473, (512, ), (1, ), 0), buf463, buf465, reinterpret_tensor(buf447, (512, ), (1, ), 0), buf437, buf439, reinterpret_tensor(buf421, (512, ), (1, ), 0), buf409, buf411, reinterpret_tensor(buf393, (512, ), (1, ), 0), buf383, buf385, reinterpret_tensor(buf367, (512, ), (1, ), 0), buf357, buf359, reinterpret_tensor(buf341, (512, ), (1, ), 0), buf331, buf333, reinterpret_tensor(buf315, (512, ), (1, ), 0), buf303, buf305, reinterpret_tensor(buf287, (512, ), (1, ), 0), buf277, buf279, reinterpret_tensor(buf261, (512, ), (1, ), 0), buf251, buf253, reinterpret_tensor(buf235, (512, ), (1, ), 0), buf225, buf227, reinterpret_tensor(buf209, (512, ), (1, ), 0), buf197, buf199, reinterpret_tensor(buf181, (512, ), (1, ), 0), buf171, buf173, reinterpret_tensor(buf155, (512, ), (1, ), 0), buf144, buf146, reinterpret_tensor(buf128, (512, ), (1, ), 0), buf117, buf119, reinterpret_tensor(buf101, (512, ), (1, ), 0), buf97, buf98, buf78, buf80, reinterpret_tensor(buf63, (1024, ), (1, ), 0), buf53, buf55, reinterpret_tensor(buf38, (1024, ), (1, ), 0), buf27, buf29, reinterpret_tensor(buf12, (1024, ), (1, ), 0), buf5, buf6, buf1015, buf1013, buf1002, buf999, reinterpret_tensor(buf987, (512, 128), (128, 1), 0), reinterpret_tensor(buf989, (512, ), (1, ), 0), reinterpret_tensor(buf982, (128, 512), (512, 1), 0), reinterpret_tensor(buf984, (128, ), (1, ), 0), buf976, buf973, reinterpret_tensor(buf961, (512, 128), (128, 1), 0), reinterpret_tensor(buf963, (512, ), (1, ), 0), reinterpret_tensor(buf956, (128, 512), (512, 1), 0), reinterpret_tensor(buf958, (128, ), (1, ), 0), buf950, buf947, reinterpret_tensor(buf935, (512, 128), (128, 1), 0), reinterpret_tensor(buf937, (512, ), (1, ), 0), reinterpret_tensor(buf930, (128, 512), (512, 1), 0), reinterpret_tensor(buf932, (128, ), (1, ), 0), buf916, buf913, buf910, buf907, reinterpret_tensor(buf894, (1024, 256), (256, 1), 0), reinterpret_tensor(buf896, (1024, ), (1, ), 0), reinterpret_tensor(buf889, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf891, (256, ), (1, ), 0), buf883, buf880, reinterpret_tensor(buf867, (1024, 256), (256, 1), 0), reinterpret_tensor(buf869, (1024, ), (1, ), 0), reinterpret_tensor(buf862, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf864, (256, ), (1, ), 0), buf856, buf853, reinterpret_tensor(buf840, (1024, 256), (256, 1), 0), reinterpret_tensor(buf842, (1024, ), (1, ), 0), reinterpret_tensor(buf835, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf837, (256, ), (1, ), 0), buf822, buf819, buf816, buf813, reinterpret_tensor(buf800, (2048, 512), (512, 1), 0), reinterpret_tensor(buf802, (2048, ), (1, ), 0), reinterpret_tensor(buf795, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf797, (512, ), (1, ), 0), buf790, buf787, reinterpret_tensor(buf774, (2048, 512), (512, 1), 0), reinterpret_tensor(buf776, (2048, ), (1, ), 0), reinterpret_tensor(buf769, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf771, (512, ), (1, ), 0), buf764, buf761, reinterpret_tensor(buf748, (2048, 512), (512, 1), 0), reinterpret_tensor(buf750, (2048, ), (1, ), 0), reinterpret_tensor(buf743, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf745, (512, ), (1, ), 0), buf736, buf733, reinterpret_tensor(buf720, (2048, 512), (512, 1), 0), reinterpret_tensor(buf722, (2048, ), (1, ), 0), reinterpret_tensor(buf715, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf717, (512, ), (1, ), 0), buf710, buf707, reinterpret_tensor(buf694, (2048, 512), (512, 1), 0), reinterpret_tensor(buf696, (2048, ), (1, ), 0), reinterpret_tensor(buf689, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf691, (512, ), (1, ), 0), buf684, buf681, reinterpret_tensor(buf668, (2048, 512), (512, 1), 0), reinterpret_tensor(buf670, (2048, ), (1, ), 0), reinterpret_tensor(buf663, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf665, (512, ), (1, ), 0), buf658, buf655, reinterpret_tensor(buf642, (2048, 512), (512, 1), 0), reinterpret_tensor(buf644, (2048, ), (1, ), 0), reinterpret_tensor(buf637, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf639, (512, ), (1, ), 0), buf630, buf627, reinterpret_tensor(buf614, (2048, 512), (512, 1), 0), reinterpret_tensor(buf616, (2048, ), (1, ), 0), reinterpret_tensor(buf609, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf611, (512, ), (1, ), 0), buf604, buf601, reinterpret_tensor(buf588, (2048, 512), (512, 1), 0), reinterpret_tensor(buf590, (2048, ), (1, ), 0), reinterpret_tensor(buf583, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf585, (512, ), (1, ), 0), buf578, buf575, reinterpret_tensor(buf562, (2048, 512), (512, 1), 0), reinterpret_tensor(buf564, (2048, ), (1, ), 0), reinterpret_tensor(buf557, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf559, (512, ), (1, ), 0), buf552, buf549, reinterpret_tensor(buf536, (2048, 512), (512, 1), 0), reinterpret_tensor(buf538, (2048, ), (1, ), 0), reinterpret_tensor(buf531, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf533, (512, ), (1, ), 0), buf524, buf521, reinterpret_tensor(buf508, (2048, 512), (512, 1), 0), reinterpret_tensor(buf510, (2048, ), (1, ), 0), reinterpret_tensor(buf503, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf505, (512, ), (1, ), 0), buf498, buf495, reinterpret_tensor(buf482, (2048, 512), (512, 1), 0), reinterpret_tensor(buf484, (2048, ), (1, ), 0), reinterpret_tensor(buf477, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf479, (512, ), (1, ), 0), buf472, buf469, reinterpret_tensor(buf456, (2048, 512), (512, 1), 0), reinterpret_tensor(buf458, (2048, ), (1, ), 0), reinterpret_tensor(buf451, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf453, (512, ), (1, ), 0), buf446, buf443, reinterpret_tensor(buf430, (2048, 512), (512, 1), 0), reinterpret_tensor(buf432, (2048, ), (1, ), 0), reinterpret_tensor(buf425, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf427, (512, ), (1, ), 0), buf418, buf415, reinterpret_tensor(buf402, (2048, 512), (512, 1), 0), reinterpret_tensor(buf404, (2048, ), (1, ), 0), reinterpret_tensor(buf397, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf399, (512, ), (1, ), 0), buf392, buf389, reinterpret_tensor(buf376, (2048, 512), (512, 1), 0), reinterpret_tensor(buf378, (2048, ), (1, ), 0), reinterpret_tensor(buf371, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf373, (512, ), (1, ), 0), buf366, buf363, reinterpret_tensor(buf350, (2048, 512), (512, 1), 0), reinterpret_tensor(buf352, (2048, ), (1, ), 0), reinterpret_tensor(buf345, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf347, (512, ), (1, ), 0), buf340, buf337, reinterpret_tensor(buf324, (2048, 512), (512, 1), 0), reinterpret_tensor(buf326, (2048, ), (1, ), 0), reinterpret_tensor(buf319, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf321, (512, ), (1, ), 0), buf312, buf309, reinterpret_tensor(buf296, (2048, 512), (512, 1), 0), reinterpret_tensor(buf298, (2048, ), (1, ), 0), reinterpret_tensor(buf291, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf293, (512, ), (1, ), 0), buf286, buf283, reinterpret_tensor(buf270, (2048, 512), (512, 1), 0), reinterpret_tensor(buf272, (2048, ), (1, ), 0), reinterpret_tensor(buf265, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf267, (512, ), (1, ), 0), buf260, buf257, reinterpret_tensor(buf244, (2048, 512), (512, 1), 0), reinterpret_tensor(buf246, (2048, ), (1, ), 0), reinterpret_tensor(buf239, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf241, (512, ), (1, ), 0), buf234, buf231, reinterpret_tensor(buf218, (2048, 512), (512, 1), 0), reinterpret_tensor(buf220, (2048, ), (1, ), 0), reinterpret_tensor(buf213, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf215, (512, ), (1, ), 0), buf206, buf203, reinterpret_tensor(buf190, (2048, 512), (512, 1), 0), reinterpret_tensor(buf192, (2048, ), (1, ), 0), reinterpret_tensor(buf185, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf187, (512, ), (1, ), 0), buf180, buf177, reinterpret_tensor(buf164, (2048, 512), (512, 1), 0), reinterpret_tensor(buf166, (2048, ), (1, ), 0), reinterpret_tensor(buf159, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf161, (512, ), (1, ), 0), buf153, buf150, reinterpret_tensor(buf137, (2048, 512), (512, 1), 0), reinterpret_tensor(buf139, (2048, ), (1, ), 0), reinterpret_tensor(buf132, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf134, (512, ), (1, ), 0), buf126, buf123, reinterpret_tensor(buf110, (2048, 512), (512, 1), 0), reinterpret_tensor(buf112, (2048, ), (1, ), 0), reinterpret_tensor(buf105, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf107, (512, ), (1, ), 0), buf92, buf89, buf87, buf84, reinterpret_tensor(buf71, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf73, (4096, ), (1, ), 0), reinterpret_tensor(buf66, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf68, (1024, ), (1, ), 0), buf62, buf59, reinterpret_tensor(buf46, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf48, (4096, ), (1, ), 0), reinterpret_tensor(buf41, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf43, (1024, ), (1, ), 0), buf36, buf33, reinterpret_tensor(buf20, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf22, (4096, ), (1, ), 0), reinterpret_tensor(buf15, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf17, (1024, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, 128, 2, 2), (512, 1, 256, 128), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, 256, 2, 2), (1024, 1, 512, 256), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((1024, 512, 2, 2), (2048, 1, 1024, 512), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda:0', dtype=torch.float32)
    view = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    add_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    mul_8 = rand_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda:0', dtype=torch.float32)
    view_10 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cuda:0', dtype=torch.float32)
    permute_15 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    mul_22 = rand_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    add_19 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    mul_28 = rand_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_9 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    add_23 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    mul_34 = rand_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cuda:0', dtype=torch.float32)
    permute_29 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_12 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_33 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_15 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_37 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_54 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_41 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_20 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_21 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_49 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_53 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_78 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_24 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_57 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_27 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_61 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_65 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_69 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_102 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_80 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_32 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_33 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_73 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_108 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_77 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_114 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_36 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_37 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_81 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_39 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_85 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_126 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_100 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_102 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_89 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_132 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_43 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_93 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_138 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_44 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_45 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_97 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_144 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_101 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_150 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_48 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_49 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_105 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_156 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_50 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_51 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_109 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_162 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_113 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_168 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_135 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_54 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_55 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_117 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_174 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_56 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_57 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_121 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_180 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_147 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_59 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_125 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_186 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_60 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_61 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_129 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_62 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_157 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_63 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    add_133 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    mul_198 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    view_160 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_162 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    addmm_65 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_204 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cuda:0', dtype=torch.float32)
    permute_139 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    mul_206 = rand_strided((8, 7, 7, 1024), (50176, 1, 7168, 7), device='cuda:0', dtype=torch.float32)
    view_165 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_66 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_67 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    add_143 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    mul_212 = rand_strided((8, 7, 7, 1024), (50176, 1, 7168, 7), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_68 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_69 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    add_147 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    mul_218 = rand_strided((8, 7, 7, 1024), (50176, 1, 7168, 7), device='cuda:0', dtype=torch.float32)
    view_175 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((392, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    addmm_71 = rand_strided((392, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_224 = rand_strided((8, 1, 1, 1024), (1024, 1, 1024, 1), device='cuda:0', dtype=torch.float32)
    clone_109 = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cuda:0', dtype=torch.float32)
    permute_182 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_198 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_248 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_298 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_304 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_308 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_314 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_328 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_334 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_338 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_358 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_364 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_374 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_384 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_424 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_428 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_444 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_458 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    permute_466 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_470 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_502 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda:0', dtype=torch.float32)
    permute_508 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_512 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_121, primals_127, primals_133, primals_139, primals_141, primals_147, primals_153, primals_159, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, primals_323, primals_325, primals_331, primals_337, primals_345, mul, permute_1, mul_2, view, addmm, view_2, addmm_1, add_5, mul_8, view_5, addmm_2, view_7, addmm_3, add_9, mul_14, view_10, addmm_4, view_12, addmm_5, mul_20, permute_15, convolution_4, mul_22, view_15, addmm_6, view_17, addmm_7, add_19, mul_28, view_20, addmm_8, view_22, addmm_9, add_23, mul_34, view_25, addmm_10, view_27, addmm_11, mul_40, permute_29, convolution_8, mul_42, view_30, addmm_12, view_32, addmm_13, add_33, mul_48, view_35, addmm_14, view_37, addmm_15, add_37, mul_54, view_40, addmm_16, view_42, addmm_17, add_41, mul_60, view_45, addmm_18, view_47, addmm_19, add_45, mul_66, view_50, addmm_20, view_52, addmm_21, add_49, mul_72, view_55, addmm_22, view_57, addmm_23, add_53, mul_78, view_60, addmm_24, view_62, addmm_25, add_57, mul_84, view_65, addmm_26, view_67, addmm_27, add_61, mul_90, view_70, addmm_28, view_72, addmm_29, add_65, mul_96, view_75, addmm_30, view_77, addmm_31, add_69, mul_102, view_80, addmm_32, view_82, addmm_33, add_73, mul_108, view_85, addmm_34, view_87, addmm_35, add_77, mul_114, view_90, addmm_36, view_92, addmm_37, add_81, mul_120, view_95, addmm_38, view_97, addmm_39, add_85, mul_126, view_100, addmm_40, view_102, addmm_41, add_89, mul_132, view_105, addmm_42, view_107, addmm_43, add_93, mul_138, view_110, addmm_44, view_112, addmm_45, add_97, mul_144, view_115, addmm_46, view_117, addmm_47, add_101, mul_150, view_120, addmm_48, view_122, addmm_49, add_105, mul_156, view_125, addmm_50, view_127, addmm_51, add_109, mul_162, view_130, addmm_52, view_132, addmm_53, add_113, mul_168, view_135, addmm_54, view_137, addmm_55, add_117, mul_174, view_140, addmm_56, view_142, addmm_57, add_121, mul_180, view_145, addmm_58, view_147, addmm_59, add_125, mul_186, view_150, addmm_60, view_152, addmm_61, add_129, mul_192, view_155, addmm_62, view_157, addmm_63, add_133, mul_198, view_160, addmm_64, view_162, addmm_65, mul_204, permute_139, convolution_36, mul_206, view_165, addmm_66, view_167, addmm_67, add_143, mul_212, view_170, addmm_68, view_172, addmm_69, add_147, mul_218, view_175, addmm_70, view_177, addmm_71, mul_224, clone_109, permute_155, div, permute_162, permute_166, div_2, permute_172, permute_176, div_3, permute_182, permute_186, div_4, div_5, permute_194, permute_198, div_6, permute_204, permute_208, div_7, permute_214, permute_218, div_8, permute_224, permute_228, div_9, permute_234, permute_238, div_10, permute_244, permute_248, div_11, permute_254, permute_258, div_12, permute_264, permute_268, div_13, permute_274, permute_278, div_14, permute_284, permute_288, div_15, permute_294, permute_298, div_16, permute_304, permute_308, div_17, permute_314, permute_318, div_18, permute_324, permute_328, div_19, permute_334, permute_338, div_20, permute_344, permute_348, div_21, permute_354, permute_358, div_22, permute_364, permute_368, div_23, permute_374, permute_378, div_24, permute_384, permute_388, div_25, permute_394, permute_398, div_26, permute_404, permute_408, div_27, permute_414, permute_418, div_28, permute_424, permute_428, div_29, permute_434, permute_438, div_30, permute_444, permute_448, div_31, permute_454, permute_458, div_32, div_33, permute_466, permute_470, div_34, permute_476, permute_480, div_35, permute_486, permute_490, div_36, div_37, permute_498, permute_502, div_38, permute_508, permute_512, div_39, permute_518, permute_522, div_40, div_41, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convnext_base', benchmark_compiled_module)
