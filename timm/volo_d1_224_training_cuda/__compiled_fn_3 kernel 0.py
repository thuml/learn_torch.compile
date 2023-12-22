
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


# kernel path: /tmp/torchinductor_youkaichao/cx/ccx7xbhxmbqep7kdwzfoge3fu7arpi53l4bjepquecwvapm2ibzw.py
# Source Nodes: [l__mod___patch_embed_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_conv_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7577cqg4n66b3xqkcmg6a3vhwdqbcq52nfscszzitfddxwo2am.py
# Source Nodes: [l__mod___patch_embed_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___patch_embed_conv_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_1', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp1 = tl.load(in_ptr1 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ylhig4hyzsq4bww3ib2lx7xey7cwthx3cxpj4dxnottbujjgx5.py
# Source Nodes: [l__mod___patch_embed_conv_1, l__mod___patch_embed_conv_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___patch_embed_conv_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# l__mod___patch_embed_conv_2 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgnab22wadsb3uf6nkk5gwrxhyjsdlxhgmfgvarusb2ew5i63tl.py
# Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm1 => clone, var_mean_3
triton_red_fused_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 2
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r3) + (75264*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyau237stxjhawwyuhia7qi47rcmlrne6txqbij5uubalatbmsw.py
# Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___network_0___0___norm1 => add_15, clone, rsqrt_3, var_mean_3
triton_per_fused_native_layer_norm_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x3), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yuzeh7zj5qotu6ryyr67h635p23v2iww7c6xeoktq7ptqm5dit.py
# Source Nodes: [getattr_l__mod___network_0___0___attn_v, getattr_l__mod___network_0___0___norm1, permute_3], Original ATen: [aten.native_layer_norm, aten.permute, aten.view]
# getattr_l__mod___network_0___0___attn_v => view
# getattr_l__mod___network_0___0___norm1 => add_15, add_16, clone, mul_21, mul_22, rsqrt_3, sub_3, var_mean_3
# permute_3 => permute_5
triton_poi_fused_native_layer_norm_permute_view_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 192.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (192*y3)), tmp15, xmask & ymask)
    tl.store(out_ptr2 + (x2 + (192*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kjurouj4tlprwcrim5e6pflfveuuqm7xmllw7umv6pduo2fhan.py
# Source Nodes: [getattr_l__mod___network_0___0___attn_unfold], Original ATen: [aten.im2col]
# getattr_l__mod___network_0___0___attn_unfold => add_17
triton_poi_fused_im2col_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_im2col_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    tmp0 = x1 + (2*x0)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebipoz6mz6spssfnuakqti4ofu5onumhsgpc6ftrlf3kftmzwpy.py
# Source Nodes: [getattr_l__mod___network_0___0___attn_attn], Original ATen: [aten.view]
# getattr_l__mod___network_0___0___attn_attn => view_4
triton_poi_fused_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*(x1 % 14)) + (10752*(x1 // 14))), None)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + (384*(x1 % 14)) + (10752*(x1 // 14))), None)
    tmp3 = tl.load(in_ptr0 + (5376 + x0 + (384*(x1 % 14)) + (10752*(x1 // 14))), None)
    tmp5 = tl.load(in_ptr0 + (5568 + x0 + (384*(x1 % 14)) + (10752*(x1 // 14))), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6ys4rk7aya2m52qxa4xmtma36qhrye3jiauxgknqrnmqzkkd35a.py
# Source Nodes: [attn_2, attn_3, attn_4], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
# attn_2 => mul_23
# attn_3 => amax, clone_2, div, exp, sub_4, sum_1
# attn_4 => clone_3
triton_per_fused__softmax_clone_detach_mul_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_mul_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 84672
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 54
    x4 = xindex % 9
    x5 = (xindex // 9) % 6
    x6 = (xindex // 54) % 196
    x7 = (xindex // 10584)
    tmp0 = tl.load(in_ptr0 + (r2 + (9*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (9*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (9*x4) + (81*x6) + (15876*x5) + (95256*x7)), tmp15, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (9*x4) + (81*x6) + (15876*x5) + (95256*x7)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr6deojgkvmgusvrkndt74uhdzw7kycc6qwetjxok2c47qpiughi.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_4
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 9
    x2 = (xindex // 288) % 196
    x0 = xindex % 32
    x3 = (xindex // 56448) % 6
    x4 = (xindex // 338688)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*(x1 // 3)) + (x2 // 14)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + ((14*(x1 % 3)) + (x2 % 14)), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 30
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 30), "index out of bounds: 0 <= tmp3 < 30")
    tmp5 = tmp4 + 30
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 30), "index out of bounds: 0 <= tmp7 < 30")
    tmp8 = (-1) + tmp3
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1], 28, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = (-1) + tmp7
    tmp14 = tmp13 >= tmp9
    tmp15 = tmp13 < tmp11
    tmp16 = tmp10 & tmp12
    tmp17 = tmp16 & tmp14
    tmp18 = tmp17 & tmp15
    tmp19 = tl.load(in_ptr1 + ((-5568) + x0 + (32*x3) + (192*tmp7) + (5376*tmp3) + (150528*x4)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tl.store(out_ptr0 + (x6), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6gtspuvsde4lmef4rz7mrldzoniynow5wr3shng5ck62ipnhge.py
# Source Nodes: [x_4], Original ATen: [aten.col2im]
# x_4 => full_default
triton_poi_fused_col2im_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1382400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j2/cj26lkyvyffdoc4eadf5w2aanqa6f6six7zo2wi2xde5ggqwygw5.py
# Source Nodes: [x_3, x_4], Original ATen: [aten.clone, aten.col2im]
# x_3 => clone_5
# x_4 => _unsafe_index_put
triton_poi_fused_clone_col2im_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_col2im_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 196
    x3 = (xindex // 196)
    y0 = yindex % 32
    y1 = (yindex // 32)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x3) + (288*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (1764*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pw/cpwn4otrm5ae6nstqkrr6l5xacuyccpcwg6zaxqhuitwgxnzy52g.py
# Source Nodes: [x_5], Original ATen: [aten._unsafe_view, aten.clone]
# x_5 => clone_6, view_12
triton_poi_fused__unsafe_view_clone_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = 1 + ((y0 // 28) % 28)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (y0 % 28)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (31 + (30*((y0 // 28) % 28)) + (900*x1) + (172800*(y0 // 784)) + (y0 % 28)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x1 + (192*y0)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnbflg5r3jqjegvx5dfhrxkbabm6of2jjlhavpjojcfa5blpqpd.py
# Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm2 => clone_8, var_mean_4
# x_5 => add_21
# x_7 => add_22
triton_red_fused_add_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 784
    x2 = (xindex // 1568)
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (784*r3) + (75264*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (96*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r3 + (96*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp9, xmask)
    tl.store(out_ptr2 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqpdlykjg7kqrqfyd6fxkjxha2ouxv7ygre7agw6j456wa5jhbo.py
# Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___network_0___0___norm2 => add_23, clone_8, rsqrt_4, var_mean_4
# x_5 => add_21
# x_7 => add_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2ddhfryx2tn2ggx65sjlzhznivj7jklryyy6dzw2rnfpkxpxdjd.py
# Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_l__mod___network_0___0___norm2 => add_23, add_24, clone_8, mul_24, mul_25, rsqrt_4, sub_5, var_mean_4
# x_5 => add_21
# x_7 => add_22
# x_8 => view_14
triton_poi_fused_add_native_layer_norm_view_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 192.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (192*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zpwvrw2o2nq2nx3evlkz56z4kz6a3rqs7mf4hv3ed2yisksf5p.py
# Source Nodes: [x_12, x_9], Original ATen: [aten.gelu, aten.view]
# x_12 => view_16
# x_9 => add_25, erf, mul_26, mul_27, mul_28
triton_poi_fused_gelu_view_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
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


# kernel path: /tmp/torchinductor_youkaichao/b5/cb53la6e2gg6onqgoxkv6udgyxuyxoqkoglkfkfrla4xsgiwgwgr.py
# Source Nodes: [getattr_l__mod___network_0___1___attn_v, getattr_l__mod___network_0___1___norm1, permute_10, x_14, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.permute, aten.view]
# getattr_l__mod___network_0___1___attn_v => view_18
# getattr_l__mod___network_0___1___norm1 => add_27, add_28, clone_11, mul_29, mul_30, rsqrt_5, sub_6, var_mean_5
# permute_10 => permute_19
# x_14 => add_26
# x_5 => add_21
# x_7 => add_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_permute_view_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_permute_view_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (150528*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (192*x3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (192*x3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp10 - tmp20
    tmp28 = 192.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 / tmp28
    tl.store(in_out_ptr0 + (r2 + (192*x3)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (192*x3)), tmp33, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (192*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (192*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr5 + (x3), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmt5wuo6shzobeblqmvbfgv3udt7llufmsdxnva7fk3clu2usgi.py
# Source Nodes: [getattr_l__mod___network_0___1___norm2, x_17, x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___network_0___1___norm2 => add_35, add_36, clone_19, mul_32, mul_33, rsqrt_6, sub_8, var_mean_6
# x_17 => add_33
# x_19 => add_34
# x_20 => view_32
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (192*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63u227gtngmlqy32v35n3njcpeu2e5wqxfpz7pjc4mmzhrygrfi.py
# Source Nodes: [getattr_l__mod___network_0___2___attn_v, getattr_l__mod___network_0___2___norm1, permute_17, x_17, x_19, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.permute, aten.view]
# getattr_l__mod___network_0___2___attn_v => view_36
# getattr_l__mod___network_0___2___norm1 => add_39, add_40, clone_22, mul_37, mul_38, rsqrt_7, sub_9, var_mean_7
# permute_17 => permute_33
# x_17 => add_33
# x_19 => add_34
# x_26 => add_38
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_permute_view_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_permute_view_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (192*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (192*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyau42fxsqnzinbqk4iwwopfi7gasp4n345x53eo5qdhes4ffwi.py
# Source Nodes: [x_41, x_43, x_51, x_52], Original ATen: [aten.add, aten.permute]
# x_41 => add_57
# x_43 => add_58
# x_51 => add_62
# x_52 => permute_57
triton_poi_fused_add_permute_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_permute_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (y0 + (784*x2) + (150528*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/76/c76wdoyben5y4wkstos3anxdjn2rj34vmc4dwhfxklzw37qfkiaa.py
# Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_2___0___norm1 => clone_45, var_mean_11
# x_56 => add_63
triton_red_fused_add_native_layer_norm_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x5 = xindex % 588
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (75264*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x6), tmp6, xmask)
    tl.store(out_ptr1 + (x6), tmp7, xmask)
    tl.store(out_ptr2 + (x6), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cumc4h4rr6wtt5w5xrtznawcwvh5mz3l5zwxjmvf6r46nwwdv3km.py
# Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___network_2___0___norm1 => add_64, clone_45, rsqrt_11, var_mean_11
# x_56 => add_63
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 384.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkjm4cxmwq4lr2qeduuzv44bpb4o32a3hqagrqjvyjf2wzdwxva.py
# Source Nodes: [getattr_l__mod___network_2___0___attn_qkv, getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_l__mod___network_2___0___attn_qkv => view_72
# getattr_l__mod___network_2___0___norm1 => add_64, add_65, clone_45, mul_53, mul_54, rsqrt_11, sub_15, var_mean_11
# x_56 => add_63
triton_poi_fused_add_native_layer_norm_view_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (384*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp13, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (384*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycexxubrflyydauvu3cvm22lukvcokxgrwsk6qmylazmusv452f.py
# Source Nodes: [matmul_4], Original ATen: [aten.clone]
# matmul_4 => clone_46
triton_poi_fused_clone_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 12
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1152*x1) + (225792*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccn7py6v3b4bdnrrv4jmtwim6r2bwptbeip5tslkd5bwykcbg7jd.py
# Source Nodes: [matmul_4], Original ATen: [aten.clone]
# matmul_4 => clone_47
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (384 + y0 + (1152*x2) + (225792*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4e7wi2azyfja3xrfpziq7t7buxsksr2gymbzr4ofnpxnbe5lfoj.py
# Source Nodes: [attn_20, attn_21, attn_22], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
# attn_20 => mul_55
# attn_21 => amax_4, div_4, exp_4, sub_16, sum_5
# attn_22 => clone_48
triton_per_fused__softmax_clone_detach_mul_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_mul_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18816
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
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (196*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujl4icsxxbigtexipgvmeuvm34tegagruso62vpefmuqtkxmp5g.py
# Source Nodes: [matmul_5], Original ATen: [aten.clone]
# matmul_5 => clone_49
triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 12
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (32*x2) + (1152*x1) + (225792*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2c7oap6edjsorpq7xai5xo4w7hmtanjcmied42r6gmlgbvw7rj.py
# Source Nodes: [x_59], Original ATen: [aten.view]
# x_59 => view_82
triton_poi_fused_view_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 196)) + (6272*(x0 // 32)) + (75264*(x1 // 196)) + (x0 % 32)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnmt4bjqjyc7ro37565dk6orvho3xew4e3qq67dj2wtac3acf4f.py
# Source Nodes: [getattr_l__mod___network_2___0___norm2, x_56, x_61, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___network_2___0___norm2 => add_67, add_68, clone_52, mul_56, mul_57, rsqrt_12, sub_17, var_mean_12
# x_56 => add_63
# x_61 => add_66
# x_62 => view_84
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (384*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6crtwzwfu46c26nqrkfmuvhs3nzr4kegjxwb4jbscfurn6hn5t.py
# Source Nodes: [x_63, x_66], Original ATen: [aten.gelu, aten.view]
# x_63 => add_69, erf_4, mul_58, mul_59, mul_60
# x_66 => view_86
triton_poi_fused_gelu_view_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
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


# kernel path: /tmp/torchinductor_youkaichao/7o/c7ochhulpzzq3kz43kk3gwucinnrpt3mf2xxg7a5makexv52g56x.py
# Source Nodes: [getattr_l__mod___network_2___1___attn_qkv, getattr_l__mod___network_2___1___norm1, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___network_2___1___attn_qkv => view_88
# getattr_l__mod___network_2___1___norm1 => add_71, add_72, clone_55, mul_61, mul_62, rsqrt_13, sub_18, var_mean_13
# x_68 => add_70
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (384*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvyrakksm3axju5uhgh7sijedrgeztovrf66dbkqimgiosrrito.py
# Source Nodes: [getattr_l__mod___network_2___1___norm2, x_68, x_72, x_73], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___network_2___1___norm2 => add_74, add_75, clone_62, mul_64, mul_65, rsqrt_14, sub_20, var_mean_14
# x_68 => add_70
# x_72 => add_73
# x_73 => view_100
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (384*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2gxguabrzbyunjgofgrtnj6tkvocr6rqu2g5j4rmd277hxs6nrz.py
# Source Nodes: [cat_5], Original ATen: [aten.cat]
# cat_5 => cat
triton_poi_fused_cat_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (384*x2) + (75648*y1)), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/cszg57j3punxbpipd3rjlqr73iskffdrx32m3iucwzybedpg7qdu.py
# Source Nodes: [l__mod___post_network_0_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___post_network_0_norm1 => add_162, add_163, mul_165, mul_166, rsqrt_39, sub_57, var_mean_39
triton_per_fused_native_layer_norm_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 384.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (384*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosshoeroma6ttdv4nhtz5ggfq2i5gdfzem5r6jzxf6z6qsd2lhi.py
# Source Nodes: [mul_18], Original ATen: [aten.mul]
# mul_18 => mul_167
triton_poi_fused_mul_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpd723qqq6sabykhiukmsx4jryzjkwvonnakfdfg3wgunwxhcdo6.py
# Source Nodes: [attn_62], Original ATen: [aten.clone]
# attn_62 => clone_185
triton_poi_fused_clone_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (151296*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvn3eg5zlbcnwrju7wmk2a4dbl6hhqcvsx4zu4qmfqhcgaatplk.py
# Source Nodes: [attn_63, attn_64], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_63 => amax_18, div_18, exp_18, sub_58, sum_19
# attn_64 => clone_186
triton_per_fused__softmax_clone_detach_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (197*x0)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (197*x0)), tmp11, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clwbxzzxazuqrsukknbnj2ialart6chathhrg6tib7n7jmrd4mws.py
# Source Nodes: [matmul_33], Original ATen: [aten.clone]
# matmul_33 => clone_187
triton_poi_fused_clone_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 197
    x2 = (xindex // 6304) % 12
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (32*x2) + (768*x1) + (151296*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5chjughnvpy6iynlct7xnnax2mau2dxlu75kv6j4y2ue4qsvbt.py
# Source Nodes: [cls_embed_4, l__mod___post_network_0_norm2, x_218], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cls_embed_4 => add_164
# l__mod___post_network_0_norm2 => add_165, add_166, mul_168, mul_169, rsqrt_40, sub_59, var_mean_40
# x_218 => view_312
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (75648*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (384*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhzgewuhy55hh6eczsdzrkranmn7ucjho2nypzsqejfn6qu45si.py
# Source Nodes: [x_219, x_222], Original ATen: [aten.gelu, aten.view]
# x_219 => add_167, erf_18, mul_170, mul_171, mul_172
# x_222 => view_314
triton_poi_fused_gelu_view_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctb473xkcprbzgkssvwkbqyeycervs3covqujfwozz23eiuybbwx.py
# Source Nodes: [cat_4, l__mod___post_network_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_4 => cat_1
# l__mod___post_network_1_norm1 => add_169, add_170, mul_173, mul_174, rsqrt_41, sub_60, var_mean_41
triton_per_fused_cat_native_layer_norm_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_41', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (75648*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (r2 + (384*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.load(in_ptr3 + (r2 + (384*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 197, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & tmp16 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 384, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = 384.0
    tmp40 = tmp38 / tmp39
    tmp41 = 1e-05
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp22 - tmp32
    tmp45 = tmp44 * tmp43
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(out_ptr0 + (r2 + (384*x3)), tmp22, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp43, xmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp49, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clg6fv3mihpw45izjzft5iurmtwt7y6eavmx3hqeqgmbrq2zvzjf.py
# Source Nodes: [aux], Original ATen: [aten._unsafe_view, aten.clone]
# aux => clone_198, view_335
triton_poi_fused__unsafe_view_clone_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (384*(x1 % 196)) + (75648*(x1 // 196))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3ba6e7kicx4kvjtxl7th6kmaapbynvofxrtcap74bon6oie4y63.py
# Source Nodes: [aux, max_1], Original ATen: [aten.add, aten.max]
# aux => add_178
# max_1 => max_1
triton_red_fused_add_max_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_max_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16000
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1000
    x1 = (xindex // 1000)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r2) + (98000*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx46sgdrgqssvndpvhjzdidrgi5kvufnmatnxfhq7txayw3gp7zd.py
# Source Nodes: [aux, max_1, mul_20, pred], Original ATen: [aten.add, aten.max, aten.mul]
# aux => add_178
# max_1 => max_1
# mul_20 => mul_183
# pred => add_179
triton_per_fused_add_max_mul_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_mul_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1000
    x1 = (xindex // 1000)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r2) + (2000*x1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp7 = tmp5 + tmp6
    tmp8 = 0.5
    tmp9 = tmp4 * tmp8
    tmp10 = tmp7 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqavo4ph7igdtuowhjmslupijbei6trdqipuncyqo5qf5lzlo6ei.py
# Source Nodes: [aux, max_1], Original ATen: [aten.add, aten.max]
# aux => add_178
# max_1 => max_1
triton_red_fused_add_max_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_max_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8000
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1000
    x1 = (xindex // 1000)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp4_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r2) + (196000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        _tmp4_next, _tmp4_index_next = triton_helpers.maximum_with_index(
            _tmp4, _tmp4_index, tmp3, rindex
        )
        _tmp4 = tl.where(rmask & xmask, _tmp4_next, _tmp4)
        _tmp4_index = tl.where(rmask & xmask, _tmp4_index_next, _tmp4_index)
    _, tmp4_tmp = triton_helpers.max_with_index(_tmp4, _tmp4_index, 1)
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/cojtfwvcneuo7qcblfm2p3dz2qnomaumkm2m3n5leswwprlmrvo6.py
# Source Nodes: [l__mod___patch_embed_conv_1], Original ATen: [aten.add]
# l__mod___patch_embed_conv_1 => add
triton_poi_fused_add_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_46', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261 = args
    args.clear()
    assert_size_stride(primals_1, (1, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(primals_2, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_3, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (192, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (192, 192), (192, 1))
    assert_size_stride(primals_17, (486, 192), (192, 1))
    assert_size_stride(primals_18, (486, ), (1, ))
    assert_size_stride(primals_19, (192, 192), (192, 1))
    assert_size_stride(primals_20, (192, ), (1, ))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_22, (192, ), (1, ))
    assert_size_stride(primals_23, (576, 192), (192, 1))
    assert_size_stride(primals_24, (576, ), (1, ))
    assert_size_stride(primals_25, (192, 576), (576, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, 192), (192, 1))
    assert_size_stride(primals_30, (486, 192), (192, 1))
    assert_size_stride(primals_31, (486, ), (1, ))
    assert_size_stride(primals_32, (192, 192), (192, 1))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_36, (576, 192), (192, 1))
    assert_size_stride(primals_37, (576, ), (1, ))
    assert_size_stride(primals_38, (192, 576), (576, 1))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, 192), (192, 1))
    assert_size_stride(primals_43, (486, 192), (192, 1))
    assert_size_stride(primals_44, (486, ), (1, ))
    assert_size_stride(primals_45, (192, 192), (192, 1))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (576, 192), (192, 1))
    assert_size_stride(primals_50, (576, ), (1, ))
    assert_size_stride(primals_51, (192, 576), (576, 1))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_54, (192, ), (1, ))
    assert_size_stride(primals_55, (192, 192), (192, 1))
    assert_size_stride(primals_56, (486, 192), (192, 1))
    assert_size_stride(primals_57, (486, ), (1, ))
    assert_size_stride(primals_58, (192, 192), (192, 1))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (576, 192), (192, 1))
    assert_size_stride(primals_63, (576, ), (1, ))
    assert_size_stride(primals_64, (192, 576), (576, 1))
    assert_size_stride(primals_65, (192, ), (1, ))
    assert_size_stride(primals_66, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (1152, 384), (384, 1))
    assert_size_stride(primals_71, (384, 384), (384, 1))
    assert_size_stride(primals_72, (384, ), (1, ))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (1152, 384), (384, 1))
    assert_size_stride(primals_76, (1152, ), (1, ))
    assert_size_stride(primals_77, (384, 1152), (1152, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_80, (384, ), (1, ))
    assert_size_stride(primals_81, (1152, 384), (384, 1))
    assert_size_stride(primals_82, (384, 384), (384, 1))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (1152, 384), (384, 1))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_88, (384, 1152), (1152, 1))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_92, (1152, 384), (384, 1))
    assert_size_stride(primals_93, (384, 384), (384, 1))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (1152, 384), (384, 1))
    assert_size_stride(primals_98, (1152, ), (1, ))
    assert_size_stride(primals_99, (384, 1152), (1152, 1))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (1152, 384), (384, 1))
    assert_size_stride(primals_104, (384, 384), (384, 1))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (1152, 384), (384, 1))
    assert_size_stride(primals_109, (1152, ), (1, ))
    assert_size_stride(primals_110, (384, 1152), (1152, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_114, (1152, 384), (384, 1))
    assert_size_stride(primals_115, (384, 384), (384, 1))
    assert_size_stride(primals_116, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (1152, 384), (384, 1))
    assert_size_stride(primals_120, (1152, ), (1, ))
    assert_size_stride(primals_121, (384, 1152), (1152, 1))
    assert_size_stride(primals_122, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (1152, 384), (384, 1))
    assert_size_stride(primals_126, (384, 384), (384, 1))
    assert_size_stride(primals_127, (384, ), (1, ))
    assert_size_stride(primals_128, (384, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (1152, 384), (384, 1))
    assert_size_stride(primals_131, (1152, ), (1, ))
    assert_size_stride(primals_132, (384, 1152), (1152, 1))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (1152, 384), (384, 1))
    assert_size_stride(primals_137, (384, 384), (384, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_141, (1152, 384), (384, 1))
    assert_size_stride(primals_142, (1152, ), (1, ))
    assert_size_stride(primals_143, (384, 1152), (1152, 1))
    assert_size_stride(primals_144, (384, ), (1, ))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (1152, 384), (384, 1))
    assert_size_stride(primals_148, (384, 384), (384, 1))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (1152, 384), (384, 1))
    assert_size_stride(primals_153, (1152, ), (1, ))
    assert_size_stride(primals_154, (384, 1152), (1152, 1))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_158, (1152, 384), (384, 1))
    assert_size_stride(primals_159, (384, 384), (384, 1))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (1152, 384), (384, 1))
    assert_size_stride(primals_164, (1152, ), (1, ))
    assert_size_stride(primals_165, (384, 1152), (1152, 1))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (1152, 384), (384, 1))
    assert_size_stride(primals_170, (384, 384), (384, 1))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (1152, 384), (384, 1))
    assert_size_stride(primals_175, (1152, ), (1, ))
    assert_size_stride(primals_176, (384, 1152), (1152, 1))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (1152, 384), (384, 1))
    assert_size_stride(primals_181, (384, 384), (384, 1))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (384, ), (1, ))
    assert_size_stride(primals_185, (1152, 384), (384, 1))
    assert_size_stride(primals_186, (1152, ), (1, ))
    assert_size_stride(primals_187, (384, 1152), (1152, 1))
    assert_size_stride(primals_188, (384, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (384, ), (1, ))
    assert_size_stride(primals_191, (1152, 384), (384, 1))
    assert_size_stride(primals_192, (384, 384), (384, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (1152, 384), (384, 1))
    assert_size_stride(primals_197, (1152, ), (1, ))
    assert_size_stride(primals_198, (384, 1152), (1152, 1))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (1152, 384), (384, 1))
    assert_size_stride(primals_203, (384, 384), (384, 1))
    assert_size_stride(primals_204, (384, ), (1, ))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (384, ), (1, ))
    assert_size_stride(primals_207, (1152, 384), (384, 1))
    assert_size_stride(primals_208, (1152, ), (1, ))
    assert_size_stride(primals_209, (384, 1152), (1152, 1))
    assert_size_stride(primals_210, (384, ), (1, ))
    assert_size_stride(primals_211, (384, ), (1, ))
    assert_size_stride(primals_212, (384, ), (1, ))
    assert_size_stride(primals_213, (1152, 384), (384, 1))
    assert_size_stride(primals_214, (384, 384), (384, 1))
    assert_size_stride(primals_215, (384, ), (1, ))
    assert_size_stride(primals_216, (384, ), (1, ))
    assert_size_stride(primals_217, (384, ), (1, ))
    assert_size_stride(primals_218, (1152, 384), (384, 1))
    assert_size_stride(primals_219, (1152, ), (1, ))
    assert_size_stride(primals_220, (384, 1152), (1152, 1))
    assert_size_stride(primals_221, (384, ), (1, ))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_224, (768, 384), (384, 1))
    assert_size_stride(primals_225, (384, 384), (384, 1))
    assert_size_stride(primals_226, (384, 384), (384, 1))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (384, ), (1, ))
    assert_size_stride(primals_229, (384, ), (1, ))
    assert_size_stride(primals_230, (1152, 384), (384, 1))
    assert_size_stride(primals_231, (1152, ), (1, ))
    assert_size_stride(primals_232, (384, 1152), (1152, 1))
    assert_size_stride(primals_233, (384, ), (1, ))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (384, ), (1, ))
    assert_size_stride(primals_236, (768, 384), (384, 1))
    assert_size_stride(primals_237, (384, 384), (384, 1))
    assert_size_stride(primals_238, (384, 384), (384, 1))
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_242, (1152, 384), (384, 1))
    assert_size_stride(primals_243, (1152, ), (1, ))
    assert_size_stride(primals_244, (384, 1152), (1152, 1))
    assert_size_stride(primals_245, (384, ), (1, ))
    assert_size_stride(primals_246, (384, ), (1, ))
    assert_size_stride(primals_247, (384, ), (1, ))
    assert_size_stride(primals_248, (1000, 384), (384, 1))
    assert_size_stride(primals_249, (1000, ), (1, ))
    assert_size_stride(primals_250, (1000, 384), (384, 1))
    assert_size_stride(primals_251, (1000, ), (1, ))
    assert_size_stride(primals_252, (64, ), (1, ))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___patch_embed_conv_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_261, primals_3, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf1 = empty_strided((1, 64, 1, 1, 13), (832, 13, 832, 832, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 64, 1, 1, 13), (832, 13, 832, 832, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 64, 1, 1, 13), (832, 13, 832, 832, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 832, 7720, grid=grid(832), stream=stream0)
        buf4 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf7 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_252, primals_253, buf4, buf5, buf7, primals_252, primals_253, 64, 13, grid=grid(64), stream=stream0)
        del primals_252
        del primals_253
        buf8 = empty((8, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_conv_1, l__mod___patch_embed_conv_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_4, primals_5, buf8, 6422528, grid=grid(6422528), stream=stream0)
        del primals_5
        # Source Nodes: [l__mod___patch_embed_conv_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf10 = buf3; del buf3  # reuse
        buf11 = buf2; del buf2  # reuse
        buf12 = buf1; del buf1  # reuse
        # Source Nodes: [l__mod___patch_embed_conv_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf9, buf10, buf11, buf12, 832, 7720, grid=grid(832), stream=stream0)
        buf13 = buf5; del buf5  # reuse
        buf14 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf16 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_conv_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf10, buf11, buf12, primals_255, primals_256, buf13, buf14, buf16, primals_255, primals_256, 64, 13, grid=grid(64), stream=stream0)
        del primals_255
        del primals_256
        buf17 = empty((8, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_conv_4, l__mod___patch_embed_conv_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf9, buf13, buf14, primals_7, primals_8, buf17, 6422528, grid=grid(6422528), stream=stream0)
        del primals_8
        # Source Nodes: [l__mod___patch_embed_conv_6], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf19 = buf12; del buf12  # reuse
        buf20 = buf11; del buf11  # reuse
        buf21 = buf10; del buf10  # reuse
        # Source Nodes: [l__mod___patch_embed_conv_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf18, buf19, buf20, buf21, 832, 7720, grid=grid(832), stream=stream0)
        buf22 = buf14; del buf14  # reuse
        buf23 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf25 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_conv_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf19, buf20, buf21, primals_258, primals_259, buf22, buf23, buf25, primals_258, primals_259, 64, 13, grid=grid(64), stream=stream0)
        del buf19
        del buf20
        del buf21
        del primals_258
        del primals_259
        buf26 = empty((8, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___patch_embed_conv_7, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf18, buf22, buf23, primals_10, primals_11, buf26, 6422528, grid=grid(6422528), stream=stream0)
        del buf23
        del primals_11
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_12, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf28 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        buf29 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_3.run(buf27, primals_13, buf28, buf29, buf30, 12544, 96, grid=grid(12544), stream=stream0)
        buf31 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf32 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf630 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_4.run(buf28, buf29, buf30, buf31, buf32, buf630, 6272, 2, grid=grid(6272), stream=stream0)
        buf34 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf35 = empty((6272, 192), device='cuda', dtype=torch.float32)
        buf38 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___attn_v, getattr_l__mod___network_0___0___norm1, permute_3], Original ATen: [aten.native_layer_norm, aten.permute, aten.view]
        triton_poi_fused_native_layer_norm_permute_view_5.run(buf27, primals_13, buf31, buf32, primals_14, primals_15, buf34, buf35, buf38, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del primals_15
        buf36 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf35, reinterpret_tensor(primals_16, (192, 192), (1, 192), 0), out=buf36)
        buf37 = empty((3, 14), device='cuda', dtype=torch.int64)
        # Source Nodes: [getattr_l__mod___network_0___0___attn_unfold], Original ATen: [aten.im2col]
        triton_poi_fused_im2col_6.run(buf37, 42, grid=grid(42), stream=stream0)
        buf39 = empty((1568, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___attn_attn], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf38, buf39, 301056, grid=grid(301056), stream=stream0)
        buf40 = empty((1568, 486), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf39, reinterpret_tensor(primals_17, (192, 486), (1, 192), 0), out=buf40)
        buf43 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        buf629 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_2, attn_3, attn_4], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_8.run(buf40, primals_18, buf43, buf629, 84672, 9, grid=grid(84672), stream=stream0)
        del primals_18
        buf44 = empty((8, 6, 196, 9, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf37, buf36, buf44, 2709504, grid=grid(2709504), stream=stream0)
        buf45 = empty((9408, 9, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf44, (9408, 9, 32), (288, 32, 1), 0), out=buf45)
        buf46 = empty((8, 192, 30, 30), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf46, 1382400, grid=grid(1382400), stream=stream0)
        buf47 = empty((8, 192, 30, 30), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf47, 1382400, grid=grid(1382400), stream=stream0)
        buf48 = empty((8, 6, 32, 9, 196), device='cuda', dtype=torch.float32)
        buf49 = reinterpret_tensor(buf48, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf48  # reuse
        # Source Nodes: [x_3, x_4], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_11.run(buf49, buf45, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        aten.index_put_(buf47, [None, None, reinterpret_tensor(buf37, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf37], buf49, True)
        buf52 = buf36; del buf36  # reuse
        # Source Nodes: [x_5], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf47, buf52, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf53 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_19, (192, 192), (1, 192), 0), out=buf53)
        buf54 = reinterpret_tensor(buf30, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf30  # reuse
        buf55 = reinterpret_tensor(buf29, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf29  # reuse
        buf56 = reinterpret_tensor(buf28, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf28  # reuse
        # Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_13.run(buf27, primals_13, buf53, primals_20, buf54, buf55, buf56, 12544, 96, grid=grid(12544), stream=stream0)
        buf57 = buf32; del buf32  # reuse
        buf58 = buf31; del buf31  # reuse
        buf628 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14.run(buf54, buf55, buf56, buf57, buf58, buf628, 6272, 2, grid=grid(6272), stream=stream0)
        del buf54
        del buf55
        del buf56
        buf60 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf61 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_15.run(buf27, primals_13, buf53, primals_20, buf57, buf58, primals_21, primals_22, buf60, buf61, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del primals_22
        buf62 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, buf61, reinterpret_tensor(primals_23, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf62)
        del primals_24
        buf63 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_16.run(buf62, buf63, 3612672, grid=grid(3612672), stream=stream0)
        buf64 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf63, reinterpret_tensor(primals_25, (576, 192), (1, 576), 0), out=buf64)
        buf65 = reinterpret_tensor(buf53, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf53  # reuse
        buf69 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf70 = empty((6272, 192), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf627 = reinterpret_tensor(buf58, (8, 28, 28, 1), (784, 28, 1, 1), 0); del buf58  # reuse
        # Source Nodes: [getattr_l__mod___network_0___1___attn_v, getattr_l__mod___network_0___1___norm1, permute_10, x_14, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.permute, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_permute_view_17.run(buf65, buf27, primals_13, primals_20, buf64, primals_26, primals_27, primals_28, buf69, buf70, buf72, buf627, 6272, 192, grid=grid(6272), stream=stream0)
        del primals_13
        del primals_20
        del primals_26
        del primals_28
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [getattr_l__mod___network_0___1___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, reinterpret_tensor(primals_29, (192, 192), (1, 192), 0), out=buf71)
        buf73 = empty((1568, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___1___attn_attn], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf72, buf73, 301056, grid=grid(301056), stream=stream0)
        buf74 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf73, reinterpret_tensor(primals_30, (192, 486), (1, 192), 0), out=buf74)
        buf77 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        buf626 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_7, attn_8, attn_9], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_8.run(buf74, primals_31, buf77, buf626, 84672, 9, grid=grid(84672), stream=stream0)
        del primals_31
        buf78 = reinterpret_tensor(buf49, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf49  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf37, buf71, buf78, 2709504, grid=grid(2709504), stream=stream0)
        buf79 = buf45; del buf45  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf78, (9408, 9, 32), (288, 32, 1), 0), out=buf79)
        buf80 = buf47; del buf47  # reuse
        # Source Nodes: [x_16], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf80, 1382400, grid=grid(1382400), stream=stream0)
        buf81 = empty((8, 6, 32, 9, 196), device='cuda', dtype=torch.float32)
        buf82 = reinterpret_tensor(buf81, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf81  # reuse
        # Source Nodes: [x_15, x_16], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_11.run(buf82, buf79, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        aten.index_put_(buf80, [None, None, reinterpret_tensor(buf37, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf37], buf82, True)
        buf85 = buf71; del buf71  # reuse
        # Source Nodes: [x_17], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf80, buf85, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf86 = reinterpret_tensor(buf27, (6272, 192), (192, 1), 0); del buf27  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf85, reinterpret_tensor(primals_32, (192, 192), (1, 192), 0), out=buf86)
        buf90 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf91 = empty((6272, 192), device='cuda', dtype=torch.float32)
        buf625 = reinterpret_tensor(buf57, (8, 28, 28, 1), (784, 28, 1, 1), 0); del buf57  # reuse
        # Source Nodes: [getattr_l__mod___network_0___1___norm2, x_17, x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18.run(buf65, buf86, primals_33, primals_34, primals_35, buf90, buf91, buf625, 6272, 192, grid=grid(6272), stream=stream0)
        del primals_35
        buf92 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, buf91, reinterpret_tensor(primals_36, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf92)
        del primals_37
        buf93 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_24], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_16.run(buf92, buf93, 3612672, grid=grid(3612672), stream=stream0)
        buf94 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf93, reinterpret_tensor(primals_38, (576, 192), (1, 576), 0), out=buf94)
        buf95 = reinterpret_tensor(buf94, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf94  # reuse
        buf99 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf100 = empty((6272, 192), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf624 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___2___attn_v, getattr_l__mod___network_0___2___norm1, permute_17, x_17, x_19, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.permute, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_permute_view_19.run(buf95, buf65, buf86, primals_33, primals_39, primals_40, primals_41, buf99, buf100, buf102, buf624, 6272, 192, grid=grid(6272), stream=stream0)
        del primals_33
        del primals_39
        del primals_41
        buf101 = buf86; del buf86  # reuse
        # Source Nodes: [getattr_l__mod___network_0___2___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_42, (192, 192), (1, 192), 0), out=buf101)
        buf103 = empty((1568, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___2___attn_attn], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf102, buf103, 301056, grid=grid(301056), stream=stream0)
        buf104 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf103, reinterpret_tensor(primals_43, (192, 486), (1, 192), 0), out=buf104)
        buf107 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        buf623 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13, attn_14], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_8.run(buf104, primals_44, buf107, buf623, 84672, 9, grid=grid(84672), stream=stream0)
        del primals_44
        buf108 = reinterpret_tensor(buf82, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf82  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf37, buf101, buf108, 2709504, grid=grid(2709504), stream=stream0)
        buf109 = buf79; del buf79  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf108, (9408, 9, 32), (288, 32, 1), 0), out=buf109)
        buf110 = buf80; del buf80  # reuse
        # Source Nodes: [x_28], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf110, 1382400, grid=grid(1382400), stream=stream0)
        buf111 = empty((8, 6, 32, 9, 196), device='cuda', dtype=torch.float32)
        buf112 = reinterpret_tensor(buf111, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf111  # reuse
        # Source Nodes: [x_27, x_28], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_11.run(buf112, buf109, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        aten.index_put_(buf110, [None, None, reinterpret_tensor(buf37, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf37], buf112, True)
        buf115 = buf101; del buf101  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf110, buf115, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf116 = reinterpret_tensor(buf65, (6272, 192), (192, 1), 0); del buf65  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf115, reinterpret_tensor(primals_45, (192, 192), (1, 192), 0), out=buf116)
        buf120 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf121 = empty((6272, 192), device='cuda', dtype=torch.float32)
        buf622 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___2___norm2, x_29, x_31, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18.run(buf95, buf116, primals_46, primals_47, primals_48, buf120, buf121, buf622, 6272, 192, grid=grid(6272), stream=stream0)
        del primals_48
        buf122 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_50, buf121, reinterpret_tensor(primals_49, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf122)
        del primals_50
        buf123 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_16.run(buf122, buf123, 3612672, grid=grid(3612672), stream=stream0)
        buf124 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf123, reinterpret_tensor(primals_51, (576, 192), (1, 576), 0), out=buf124)
        buf125 = reinterpret_tensor(buf124, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf124  # reuse
        buf129 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf130 = empty((6272, 192), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf621 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___3___attn_v, getattr_l__mod___network_0___3___norm1, permute_24, x_29, x_31, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.permute, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_permute_view_19.run(buf125, buf95, buf116, primals_46, primals_52, primals_53, primals_54, buf129, buf130, buf132, buf621, 6272, 192, grid=grid(6272), stream=stream0)
        del primals_46
        del primals_52
        del primals_54
        buf131 = reinterpret_tensor(buf95, (6272, 192), (192, 1), 0); del buf95  # reuse
        # Source Nodes: [getattr_l__mod___network_0___3___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf130, reinterpret_tensor(primals_55, (192, 192), (1, 192), 0), out=buf131)
        buf133 = empty((1568, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___3___attn_attn], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf132, buf133, 301056, grid=grid(301056), stream=stream0)
        buf134 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf133, reinterpret_tensor(primals_56, (192, 486), (1, 192), 0), out=buf134)
        buf137 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        buf620 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_17, attn_18, attn_19], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_8.run(buf134, primals_57, buf137, buf620, 84672, 9, grid=grid(84672), stream=stream0)
        del buf134
        del primals_57
        buf138 = reinterpret_tensor(buf112, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf112  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf37, buf131, buf138, 2709504, grid=grid(2709504), stream=stream0)
        buf139 = buf109; del buf109  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf138, (9408, 9, 32), (288, 32, 1), 0), out=buf139)
        buf140 = buf110; del buf110  # reuse
        # Source Nodes: [x_40], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf140, 1382400, grid=grid(1382400), stream=stream0)
        buf141 = empty((8, 6, 32, 9, 196), device='cuda', dtype=torch.float32)
        buf142 = reinterpret_tensor(buf141, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf141  # reuse
        # Source Nodes: [x_39, x_40], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_11.run(buf142, buf139, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        del buf139
        aten.index_put_(buf140, [None, None, reinterpret_tensor(buf37, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf37], buf142, True)
        del buf142
        buf145 = buf131; del buf131  # reuse
        # Source Nodes: [x_41], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_12.run(buf140, buf145, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del buf140
        buf146 = buf116; del buf116  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.mm]
        extern_kernels.mm(buf145, reinterpret_tensor(primals_58, (192, 192), (1, 192), 0), out=buf146)
        buf150 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf151 = empty((6272, 192), device='cuda', dtype=torch.float32)
        buf619 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___3___norm2, x_41, x_43, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_18.run(buf125, buf146, primals_59, primals_60, primals_61, buf150, buf151, buf619, 6272, 192, grid=grid(6272), stream=stream0)
        del primals_61
        buf152 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_63, buf151, reinterpret_tensor(primals_62, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf152)
        del primals_63
        buf153 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_48], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_16.run(buf152, buf153, 3612672, grid=grid(3612672), stream=stream0)
        buf154 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf153, reinterpret_tensor(primals_64, (576, 192), (1, 576), 0), out=buf154)
        buf155 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41, x_43, x_51, x_52], Original ATen: [aten.add, aten.permute]
        triton_poi_fused_add_permute_20.run(buf125, buf146, primals_59, buf154, primals_65, buf155, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del buf125
        del buf146
        del buf154
        del primals_59
        del primals_65
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_66, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf157 = empty_strided((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), device='cuda', dtype=torch.float32)
        buf158 = empty_strided((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), device='cuda', dtype=torch.float32)
        buf159 = empty_strided((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_21.run(buf156, primals_67, primals_1, buf157, buf158, buf159, 4704, 128, grid=grid(4704), stream=stream0)
        buf160 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        buf161 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        buf618 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf157, buf158, buf159, buf160, buf161, buf618, 1568, 3, grid=grid(1568), stream=stream0)
        del buf157
        del buf158
        del buf159
        buf163 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf164 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___attn_qkv, getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_23.run(buf156, primals_67, primals_1, buf160, buf161, primals_68, primals_69, buf163, buf164, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_69
        buf165 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf164, reinterpret_tensor(primals_70, (384, 1152), (1, 384), 0), out=buf165)
        buf166 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf165, buf166, 602112, grid=grid(602112), stream=stream0)
        buf167 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf165, buf167, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf168 = empty((96, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf167, (96, 32, 196), (6272, 196, 1), 0), out=buf168)
        buf171 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf617 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20, attn_21, attn_22], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf168, buf171, buf617, 18816, 196, grid=grid(18816), stream=stream0)
        buf172 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf165, buf172, 602112, grid=grid(602112), stream=stream0)
        buf173 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf172, (96, 196, 32), (6272, 32, 1), 0), out=buf173)
        buf174 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf173, buf174, 602112, grid=grid(602112), stream=stream0)
        buf175 = reinterpret_tensor(buf173, (1568, 384), (384, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf174, reinterpret_tensor(primals_71, (384, 384), (1, 384), 0), out=buf175)
        buf176 = reinterpret_tensor(buf175, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf175  # reuse
        buf180 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf181 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf616 = reinterpret_tensor(buf161, (8, 14, 14, 1), (196, 14, 1, 1), 0); del buf161  # reuse
        # Source Nodes: [getattr_l__mod___network_2___0___norm2, x_56, x_61, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_29.run(buf176, buf156, primals_67, primals_1, primals_72, primals_73, primals_74, buf180, buf181, buf616, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_1
        del primals_67
        del primals_72
        del primals_74
        buf182 = buf165; del buf165  # reuse
        # Source Nodes: [x_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_76, buf181, reinterpret_tensor(primals_75, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf182)
        del primals_76
        buf183 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_66], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf182, buf183, 1806336, grid=grid(1806336), stream=stream0)
        buf184 = reinterpret_tensor(buf156, (1568, 384), (384, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf183, reinterpret_tensor(primals_77, (1152, 384), (1, 1152), 0), out=buf184)
        buf188 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf189 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf615 = reinterpret_tensor(buf160, (8, 14, 14, 1), (196, 14, 1, 1), 0); del buf160  # reuse
        # Source Nodes: [getattr_l__mod___network_2___1___attn_qkv, getattr_l__mod___network_2___1___norm1, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf176, buf184, primals_78, primals_79, primals_80, buf188, buf189, buf615, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_80
        buf190 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___1___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf189, reinterpret_tensor(primals_81, (384, 1152), (1, 384), 0), out=buf190)
        buf191 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf190, buf191, 602112, grid=grid(602112), stream=stream0)
        buf192 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf190, buf192, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf193 = buf168; del buf168  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf192, (96, 32, 196), (6272, 196, 1), 0), out=buf193)
        buf196 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf614 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_23, attn_24, attn_25], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf193, buf196, buf614, 18816, 196, grid=grid(18816), stream=stream0)
        buf197 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf190, buf197, 602112, grid=grid(602112), stream=stream0)
        buf198 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf196, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf197, (96, 196, 32), (6272, 32, 1), 0), out=buf198)
        buf199 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf198, buf199, 602112, grid=grid(602112), stream=stream0)
        buf200 = reinterpret_tensor(buf198, (1568, 384), (384, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf199, reinterpret_tensor(primals_82, (384, 384), (1, 384), 0), out=buf200)
        buf201 = reinterpret_tensor(buf200, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf200  # reuse
        buf205 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf206 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf613 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___1___norm2, x_68, x_72, x_73], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf201, buf176, buf184, primals_78, primals_83, primals_84, primals_85, buf205, buf206, buf613, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_78
        del primals_83
        del primals_85
        buf207 = buf190; del buf190  # reuse
        # Source Nodes: [x_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_87, buf206, reinterpret_tensor(primals_86, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf207)
        del primals_87
        buf208 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_77], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf207, buf208, 1806336, grid=grid(1806336), stream=stream0)
        buf209 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf208, reinterpret_tensor(primals_88, (1152, 384), (1, 1152), 0), out=buf209)
        buf213 = buf176; del buf176  # reuse
        buf214 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf612 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___2___attn_qkv, getattr_l__mod___network_2___2___norm1, x_79], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf201, buf209, primals_89, primals_90, primals_91, buf213, buf214, buf612, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_91
        buf215 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___2___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf214, reinterpret_tensor(primals_92, (384, 1152), (1, 384), 0), out=buf215)
        buf216 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf215, buf216, 602112, grid=grid(602112), stream=stream0)
        buf217 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf215, buf217, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf218 = buf193; del buf193  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf217, (96, 32, 196), (6272, 196, 1), 0), out=buf218)
        buf221 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf611 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_26, attn_27, attn_28], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf218, buf221, buf611, 18816, 196, grid=grid(18816), stream=stream0)
        buf222 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf215, buf222, 602112, grid=grid(602112), stream=stream0)
        buf223 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf221, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf222, (96, 196, 32), (6272, 32, 1), 0), out=buf223)
        buf224 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf223, buf224, 602112, grid=grid(602112), stream=stream0)
        buf225 = reinterpret_tensor(buf223, (1568, 384), (384, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf224, reinterpret_tensor(primals_93, (384, 384), (1, 384), 0), out=buf225)
        buf226 = reinterpret_tensor(buf225, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf225  # reuse
        buf230 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf231 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf610 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___2___norm2, x_79, x_83, x_84], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf226, buf201, buf209, primals_89, primals_94, primals_95, primals_96, buf230, buf231, buf610, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_89
        del primals_94
        del primals_96
        buf232 = buf215; del buf215  # reuse
        # Source Nodes: [x_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_98, buf231, reinterpret_tensor(primals_97, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf232)
        del primals_98
        buf233 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85, x_88], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf232, buf233, 1806336, grid=grid(1806336), stream=stream0)
        buf234 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf233, reinterpret_tensor(primals_99, (1152, 384), (1, 1152), 0), out=buf234)
        buf238 = buf201; del buf201  # reuse
        buf239 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf609 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___3___attn_qkv, getattr_l__mod___network_2___3___norm1, x_90], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf226, buf234, primals_100, primals_101, primals_102, buf238, buf239, buf609, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_102
        buf240 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___3___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf239, reinterpret_tensor(primals_103, (384, 1152), (1, 384), 0), out=buf240)
        buf241 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf240, buf241, 602112, grid=grid(602112), stream=stream0)
        buf242 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf240, buf242, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf243 = buf218; del buf218  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf242, (96, 32, 196), (6272, 196, 1), 0), out=buf243)
        buf246 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf608 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_29, attn_30, attn_31], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf243, buf246, buf608, 18816, 196, grid=grid(18816), stream=stream0)
        buf247 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf240, buf247, 602112, grid=grid(602112), stream=stream0)
        buf248 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf247, (96, 196, 32), (6272, 32, 1), 0), out=buf248)
        buf249 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf248, buf249, 602112, grid=grid(602112), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (1568, 384), (384, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf249, reinterpret_tensor(primals_104, (384, 384), (1, 384), 0), out=buf250)
        buf251 = reinterpret_tensor(buf250, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf250  # reuse
        buf255 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf256 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf607 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___3___norm2, x_90, x_94, x_95], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf251, buf226, buf234, primals_100, primals_105, primals_106, primals_107, buf255, buf256, buf607, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_100
        del primals_105
        del primals_107
        buf257 = buf240; del buf240  # reuse
        # Source Nodes: [x_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_109, buf256, reinterpret_tensor(primals_108, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf257)
        del primals_109
        buf258 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96, x_99], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf257, buf258, 1806336, grid=grid(1806336), stream=stream0)
        buf259 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf258, reinterpret_tensor(primals_110, (1152, 384), (1, 1152), 0), out=buf259)
        buf263 = buf226; del buf226  # reuse
        buf264 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf606 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___0___attn_qkv, getattr_l__mod___network_3___0___norm1, x_102], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf251, buf259, primals_111, primals_112, primals_113, buf263, buf264, buf606, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_113
        buf265 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___0___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf264, reinterpret_tensor(primals_114, (384, 1152), (1, 384), 0), out=buf265)
        buf266 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf265, buf266, 602112, grid=grid(602112), stream=stream0)
        buf267 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf265, buf267, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf268 = buf243; del buf243  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf267, (96, 32, 196), (6272, 196, 1), 0), out=buf268)
        buf271 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf605 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_32, attn_33, attn_34], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf268, buf271, buf605, 18816, 196, grid=grid(18816), stream=stream0)
        buf272 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf265, buf272, 602112, grid=grid(602112), stream=stream0)
        buf273 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf271, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf272, (96, 196, 32), (6272, 32, 1), 0), out=buf273)
        buf274 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf273, buf274, 602112, grid=grid(602112), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (1568, 384), (384, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf274, reinterpret_tensor(primals_115, (384, 384), (1, 384), 0), out=buf275)
        buf276 = reinterpret_tensor(buf275, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf275  # reuse
        buf280 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf281 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf604 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___0___norm2, x_102, x_106, x_107], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf276, buf251, buf259, primals_111, primals_116, primals_117, primals_118, buf280, buf281, buf604, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_111
        del primals_116
        del primals_118
        buf282 = buf265; del buf265  # reuse
        # Source Nodes: [x_107], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf281, reinterpret_tensor(primals_119, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf282)
        del primals_120
        buf283 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108, x_111], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf282, buf283, 1806336, grid=grid(1806336), stream=stream0)
        buf284 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf283, reinterpret_tensor(primals_121, (1152, 384), (1, 1152), 0), out=buf284)
        buf288 = buf251; del buf251  # reuse
        buf289 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf603 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___1___attn_qkv, getattr_l__mod___network_3___1___norm1, x_113], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf276, buf284, primals_122, primals_123, primals_124, buf288, buf289, buf603, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_124
        buf290 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___1___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf289, reinterpret_tensor(primals_125, (384, 1152), (1, 384), 0), out=buf290)
        buf291 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf290, buf291, 602112, grid=grid(602112), stream=stream0)
        buf292 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf290, buf292, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf293 = buf268; del buf268  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf292, (96, 32, 196), (6272, 196, 1), 0), out=buf293)
        buf296 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf602 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_35, attn_36, attn_37], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf293, buf296, buf602, 18816, 196, grid=grid(18816), stream=stream0)
        buf297 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf290, buf297, 602112, grid=grid(602112), stream=stream0)
        buf298 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf297, (96, 196, 32), (6272, 32, 1), 0), out=buf298)
        buf299 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf298, buf299, 602112, grid=grid(602112), stream=stream0)
        buf300 = reinterpret_tensor(buf298, (1568, 384), (384, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf299, reinterpret_tensor(primals_126, (384, 384), (1, 384), 0), out=buf300)
        buf301 = reinterpret_tensor(buf300, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf300  # reuse
        buf305 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf306 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf601 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___1___norm2, x_113, x_117, x_118], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf301, buf276, buf284, primals_122, primals_127, primals_128, primals_129, buf305, buf306, buf601, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_122
        del primals_127
        del primals_129
        buf307 = buf290; del buf290  # reuse
        # Source Nodes: [x_118], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_131, buf306, reinterpret_tensor(primals_130, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf307)
        del primals_131
        buf308 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_122], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf307, buf308, 1806336, grid=grid(1806336), stream=stream0)
        buf309 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf308, reinterpret_tensor(primals_132, (1152, 384), (1, 1152), 0), out=buf309)
        buf313 = buf276; del buf276  # reuse
        buf314 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf600 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___2___attn_qkv, getattr_l__mod___network_3___2___norm1, x_124], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf301, buf309, primals_133, primals_134, primals_135, buf313, buf314, buf600, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_135
        buf315 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___2___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf314, reinterpret_tensor(primals_136, (384, 1152), (1, 384), 0), out=buf315)
        buf316 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf315, buf316, 602112, grid=grid(602112), stream=stream0)
        buf317 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf315, buf317, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf318 = buf293; del buf293  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf317, (96, 32, 196), (6272, 196, 1), 0), out=buf318)
        buf321 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf599 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_38, attn_39, attn_40], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf318, buf321, buf599, 18816, 196, grid=grid(18816), stream=stream0)
        buf322 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf315, buf322, 602112, grid=grid(602112), stream=stream0)
        buf323 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf322, (96, 196, 32), (6272, 32, 1), 0), out=buf323)
        buf324 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf323, buf324, 602112, grid=grid(602112), stream=stream0)
        buf325 = reinterpret_tensor(buf323, (1568, 384), (384, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf324, reinterpret_tensor(primals_137, (384, 384), (1, 384), 0), out=buf325)
        buf326 = reinterpret_tensor(buf325, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf325  # reuse
        buf330 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf331 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf598 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___2___norm2, x_124, x_128, x_129], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf326, buf301, buf309, primals_133, primals_138, primals_139, primals_140, buf330, buf331, buf598, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_133
        del primals_138
        del primals_140
        buf332 = buf315; del buf315  # reuse
        # Source Nodes: [x_129], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_142, buf331, reinterpret_tensor(primals_141, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf332)
        del primals_142
        buf333 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130, x_133], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf332, buf333, 1806336, grid=grid(1806336), stream=stream0)
        buf334 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf333, reinterpret_tensor(primals_143, (1152, 384), (1, 1152), 0), out=buf334)
        buf338 = buf301; del buf301  # reuse
        buf339 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf597 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___3___attn_qkv, getattr_l__mod___network_3___3___norm1, x_135], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf326, buf334, primals_144, primals_145, primals_146, buf338, buf339, buf597, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_146
        buf340 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___3___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf339, reinterpret_tensor(primals_147, (384, 1152), (1, 384), 0), out=buf340)
        buf341 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf340, buf341, 602112, grid=grid(602112), stream=stream0)
        buf342 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf340, buf342, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf343 = buf318; del buf318  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf341, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf342, (96, 32, 196), (6272, 196, 1), 0), out=buf343)
        buf346 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf596 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_41, attn_42, attn_43], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf343, buf346, buf596, 18816, 196, grid=grid(18816), stream=stream0)
        buf347 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf340, buf347, 602112, grid=grid(602112), stream=stream0)
        buf348 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf346, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf347, (96, 196, 32), (6272, 32, 1), 0), out=buf348)
        buf349 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf348, buf349, 602112, grid=grid(602112), stream=stream0)
        buf350 = reinterpret_tensor(buf348, (1568, 384), (384, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf349, reinterpret_tensor(primals_148, (384, 384), (1, 384), 0), out=buf350)
        buf351 = reinterpret_tensor(buf350, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf350  # reuse
        buf355 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf356 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf595 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___3___norm2, x_135, x_139, x_140], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf351, buf326, buf334, primals_144, primals_149, primals_150, primals_151, buf355, buf356, buf595, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_144
        del primals_149
        del primals_151
        buf357 = buf340; del buf340  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_153, buf356, reinterpret_tensor(primals_152, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf357)
        del primals_153
        buf358 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141, x_144], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf357, buf358, 1806336, grid=grid(1806336), stream=stream0)
        buf359 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf358, reinterpret_tensor(primals_154, (1152, 384), (1, 1152), 0), out=buf359)
        buf363 = buf326; del buf326  # reuse
        buf364 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf594 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___4___attn_qkv, getattr_l__mod___network_3___4___norm1, x_146], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf351, buf359, primals_155, primals_156, primals_157, buf363, buf364, buf594, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_157
        buf365 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___4___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf364, reinterpret_tensor(primals_158, (384, 1152), (1, 384), 0), out=buf365)
        buf366 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf365, buf366, 602112, grid=grid(602112), stream=stream0)
        buf367 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf365, buf367, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf368 = buf343; del buf343  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf367, (96, 32, 196), (6272, 196, 1), 0), out=buf368)
        buf371 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf593 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_44, attn_45, attn_46], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf368, buf371, buf593, 18816, 196, grid=grid(18816), stream=stream0)
        buf372 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf365, buf372, 602112, grid=grid(602112), stream=stream0)
        buf373 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf371, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf372, (96, 196, 32), (6272, 32, 1), 0), out=buf373)
        buf374 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf373, buf374, 602112, grid=grid(602112), stream=stream0)
        buf375 = reinterpret_tensor(buf373, (1568, 384), (384, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf374, reinterpret_tensor(primals_159, (384, 384), (1, 384), 0), out=buf375)
        buf376 = reinterpret_tensor(buf375, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf375  # reuse
        buf380 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf381 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf592 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___4___norm2, x_146, x_150, x_151], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf376, buf351, buf359, primals_155, primals_160, primals_161, primals_162, buf380, buf381, buf592, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_155
        del primals_160
        del primals_162
        buf382 = buf365; del buf365  # reuse
        # Source Nodes: [x_151], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_164, buf381, reinterpret_tensor(primals_163, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf382)
        del primals_164
        buf383 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152, x_155], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf382, buf383, 1806336, grid=grid(1806336), stream=stream0)
        buf384 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf383, reinterpret_tensor(primals_165, (1152, 384), (1, 1152), 0), out=buf384)
        buf388 = buf351; del buf351  # reuse
        buf389 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf591 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___5___attn_qkv, getattr_l__mod___network_3___5___norm1, x_157], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf376, buf384, primals_166, primals_167, primals_168, buf388, buf389, buf591, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_168
        buf390 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___5___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf389, reinterpret_tensor(primals_169, (384, 1152), (1, 384), 0), out=buf390)
        buf391 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf390, buf391, 602112, grid=grid(602112), stream=stream0)
        buf392 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf390, buf392, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf393 = buf368; del buf368  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf391, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf392, (96, 32, 196), (6272, 196, 1), 0), out=buf393)
        buf396 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf590 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_47, attn_48, attn_49], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf393, buf396, buf590, 18816, 196, grid=grid(18816), stream=stream0)
        buf397 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf390, buf397, 602112, grid=grid(602112), stream=stream0)
        buf398 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf397, (96, 196, 32), (6272, 32, 1), 0), out=buf398)
        buf399 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf398, buf399, 602112, grid=grid(602112), stream=stream0)
        buf400 = reinterpret_tensor(buf398, (1568, 384), (384, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf399, reinterpret_tensor(primals_170, (384, 384), (1, 384), 0), out=buf400)
        buf401 = reinterpret_tensor(buf400, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf400  # reuse
        buf405 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf406 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf589 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___5___norm2, x_157, x_161, x_162], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf401, buf376, buf384, primals_166, primals_171, primals_172, primals_173, buf405, buf406, buf589, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_166
        del primals_171
        del primals_173
        buf407 = buf390; del buf390  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_175, buf406, reinterpret_tensor(primals_174, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf407)
        del primals_175
        buf408 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163, x_166], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf407, buf408, 1806336, grid=grid(1806336), stream=stream0)
        buf409 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf408, reinterpret_tensor(primals_176, (1152, 384), (1, 1152), 0), out=buf409)
        buf413 = buf376; del buf376  # reuse
        buf414 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf588 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___6___attn_qkv, getattr_l__mod___network_3___6___norm1, x_168], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf401, buf409, primals_177, primals_178, primals_179, buf413, buf414, buf588, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_179
        buf415 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___6___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf414, reinterpret_tensor(primals_180, (384, 1152), (1, 384), 0), out=buf415)
        buf416 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf415, buf416, 602112, grid=grid(602112), stream=stream0)
        buf417 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf415, buf417, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf418 = buf393; del buf393  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf416, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf417, (96, 32, 196), (6272, 196, 1), 0), out=buf418)
        buf421 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf587 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_50, attn_51, attn_52], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf418, buf421, buf587, 18816, 196, grid=grid(18816), stream=stream0)
        buf422 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf415, buf422, 602112, grid=grid(602112), stream=stream0)
        buf423 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf422, (96, 196, 32), (6272, 32, 1), 0), out=buf423)
        buf424 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf423, buf424, 602112, grid=grid(602112), stream=stream0)
        buf425 = reinterpret_tensor(buf423, (1568, 384), (384, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf424, reinterpret_tensor(primals_181, (384, 384), (1, 384), 0), out=buf425)
        buf426 = reinterpret_tensor(buf425, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf425  # reuse
        buf430 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf431 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf586 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___6___norm2, x_168, x_172, x_173], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf426, buf401, buf409, primals_177, primals_182, primals_183, primals_184, buf430, buf431, buf586, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_177
        del primals_182
        del primals_184
        buf432 = buf415; del buf415  # reuse
        # Source Nodes: [x_173], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_186, buf431, reinterpret_tensor(primals_185, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf432)
        del primals_186
        buf433 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174, x_177], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf432, buf433, 1806336, grid=grid(1806336), stream=stream0)
        buf434 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf433, reinterpret_tensor(primals_187, (1152, 384), (1, 1152), 0), out=buf434)
        buf438 = buf401; del buf401  # reuse
        buf439 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf585 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___7___attn_qkv, getattr_l__mod___network_3___7___norm1, x_179], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf426, buf434, primals_188, primals_189, primals_190, buf438, buf439, buf585, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_190
        buf440 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___7___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf439, reinterpret_tensor(primals_191, (384, 1152), (1, 384), 0), out=buf440)
        buf441 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf440, buf441, 602112, grid=grid(602112), stream=stream0)
        buf442 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf440, buf442, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf443 = buf418; del buf418  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf442, (96, 32, 196), (6272, 196, 1), 0), out=buf443)
        buf446 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf584 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_53, attn_54, attn_55], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf443, buf446, buf584, 18816, 196, grid=grid(18816), stream=stream0)
        buf447 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf440, buf447, 602112, grid=grid(602112), stream=stream0)
        buf448 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf446, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf447, (96, 196, 32), (6272, 32, 1), 0), out=buf448)
        buf449 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf448, buf449, 602112, grid=grid(602112), stream=stream0)
        buf450 = reinterpret_tensor(buf448, (1568, 384), (384, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf449, reinterpret_tensor(primals_192, (384, 384), (1, 384), 0), out=buf450)
        buf451 = reinterpret_tensor(buf450, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf450  # reuse
        buf455 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf456 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf583 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_3___7___norm2, x_179, x_183, x_184], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf451, buf426, buf434, primals_188, primals_193, primals_194, primals_195, buf455, buf456, buf583, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_188
        del primals_193
        del primals_195
        buf457 = buf440; del buf440  # reuse
        # Source Nodes: [x_184], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_197, buf456, reinterpret_tensor(primals_196, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf457)
        del primals_197
        buf458 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_185, x_188], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf457, buf458, 1806336, grid=grid(1806336), stream=stream0)
        buf459 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf458, reinterpret_tensor(primals_198, (1152, 384), (1, 1152), 0), out=buf459)
        buf463 = buf426; del buf426  # reuse
        buf464 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf582 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_4___0___attn_qkv, getattr_l__mod___network_4___0___norm1, x_191], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf451, buf459, primals_199, primals_200, primals_201, buf463, buf464, buf582, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_201
        buf465 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_4___0___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf464, reinterpret_tensor(primals_202, (384, 1152), (1, 384), 0), out=buf465)
        buf466 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf465, buf466, 602112, grid=grid(602112), stream=stream0)
        buf467 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf465, buf467, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf468 = buf443; del buf443  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf466, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf467, (96, 32, 196), (6272, 196, 1), 0), out=buf468)
        buf471 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf581 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_56, attn_57, attn_58], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf468, buf471, buf581, 18816, 196, grid=grid(18816), stream=stream0)
        buf472 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf465, buf472, 602112, grid=grid(602112), stream=stream0)
        buf473 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf471, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf472, (96, 196, 32), (6272, 32, 1), 0), out=buf473)
        buf474 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf473, buf474, 602112, grid=grid(602112), stream=stream0)
        buf475 = reinterpret_tensor(buf473, (1568, 384), (384, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf474, reinterpret_tensor(primals_203, (384, 384), (1, 384), 0), out=buf475)
        buf476 = reinterpret_tensor(buf475, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf475  # reuse
        buf480 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf481 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf580 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_4___0___norm2, x_191, x_195, x_196], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf476, buf451, buf459, primals_199, primals_204, primals_205, primals_206, buf480, buf481, buf580, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_199
        del primals_204
        del primals_206
        buf482 = buf465; del buf465  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_208, buf481, reinterpret_tensor(primals_207, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf482)
        del primals_208
        buf483 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197, x_200], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf482, buf483, 1806336, grid=grid(1806336), stream=stream0)
        buf484 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf483, reinterpret_tensor(primals_209, (1152, 384), (1, 1152), 0), out=buf484)
        buf488 = buf451; del buf451  # reuse
        buf489 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf579 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_4___1___attn_qkv, getattr_l__mod___network_4___1___norm1, x_202], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_31.run(buf476, buf484, primals_210, primals_211, primals_212, buf488, buf489, buf579, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_212
        buf490 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_4___1___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf489, reinterpret_tensor(primals_213, (384, 1152), (1, 384), 0), out=buf490)
        buf491 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf490, buf491, 602112, grid=grid(602112), stream=stream0)
        buf492 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf490, buf492, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf493 = buf468; del buf468  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf491, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf492, (96, 32, 196), (6272, 196, 1), 0), out=buf493)
        buf496 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        buf578 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_59, attn_60, attn_61], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf493, buf496, buf578, 18816, 196, grid=grid(18816), stream=stream0)
        del buf493
        buf497 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf490, buf497, 602112, grid=grid(602112), stream=stream0)
        buf498 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf497, (96, 196, 32), (6272, 32, 1), 0), out=buf498)
        buf499 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.view]
        triton_poi_fused_view_28.run(buf498, buf499, 602112, grid=grid(602112), stream=stream0)
        buf500 = reinterpret_tensor(buf498, (1568, 384), (384, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf499, reinterpret_tensor(primals_214, (384, 384), (1, 384), 0), out=buf500)
        buf501 = reinterpret_tensor(buf500, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf500  # reuse
        buf505 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        buf506 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf577 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_4___1___norm2, x_202, x_206, x_207], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_32.run(buf501, buf476, buf484, primals_210, primals_215, primals_216, primals_217, buf505, buf506, buf577, 1568, 384, grid=grid(1568), stream=stream0)
        del buf476
        del primals_210
        del primals_215
        del primals_217
        buf507 = buf490; del buf490  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_219, buf506, reinterpret_tensor(primals_218, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf507)
        del primals_219
        buf508 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_208, x_211], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_30.run(buf507, buf508, 1806336, grid=grid(1806336), stream=stream0)
        buf509 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf508, reinterpret_tensor(primals_220, (1152, 384), (1, 1152), 0), out=buf509)
        buf510 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_33.run(primals_2, buf501, buf509, primals_221, buf510, 3072, 197, grid=grid(3072, 197), stream=stream0)
        del buf501
        del primals_2
        del primals_221
        buf511 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf512 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf514 = reinterpret_tensor(buf512, (8, 197, 1), (197, 1, 1), 0); del buf512  # reuse
        buf515 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_0_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_34.run(buf514, buf510, primals_222, primals_223, buf511, buf515, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_223
        buf516 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_0_attn_kv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (1576, 384), (384, 1), 0), reinterpret_tensor(primals_224, (384, 768), (1, 384), 0), out=buf516)
        buf517 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_0_attn_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (8, 384), (75648, 1), 0), reinterpret_tensor(primals_225, (384, 384), (1, 384), 0), out=buf517)
        buf518 = reinterpret_tensor(buf517, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf517  # reuse
        # Source Nodes: [mul_18], Original ATen: [aten.mul]
        triton_poi_fused_mul_35.run(buf518, 3072, grid=grid(3072), stream=stream0)
        buf519 = empty((8, 12, 32, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf516, buf519, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf520 = empty((96, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf518, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf519, (96, 32, 197), (6304, 197, 1), 0), out=buf520)
        buf523 = empty((8, 12, 1, 197), device='cuda', dtype=torch.float32)
        buf576 = empty((8, 12, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_63, attn_64], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_37.run(buf520, buf523, buf576, 96, 197, grid=grid(96), stream=stream0)
        buf524 = empty((8, 12, 197, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf516, buf524, 605184, grid=grid(605184), stream=stream0)
        buf525 = empty((96, 1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf523, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf524, (96, 197, 32), (6304, 32, 1), 0), out=buf525)
        buf526 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf525, (8, 384), (384, 1), 0), reinterpret_tensor(primals_226, (384, 384), (1, 384), 0), out=buf526)
        buf530 = empty((8, 1, 384), device='cuda', dtype=torch.float32)
        buf531 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf575 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_embed_4, l__mod___post_network_0_norm2, x_218], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_39.run(buf510, buf526, primals_227, primals_228, primals_229, buf530, buf531, buf575, 8, 384, grid=grid(8), stream=stream0)
        del primals_229
        buf532 = empty((8, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_231, buf531, reinterpret_tensor(primals_230, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf532)
        del primals_231
        buf533 = empty((8, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219, x_222], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf532, buf533, 9216, grid=grid(9216), stream=stream0)
        buf534 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf533, reinterpret_tensor(primals_232, (1152, 384), (1, 1152), 0), out=buf534)
        buf535 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf536 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf537 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf539 = reinterpret_tensor(buf537, (8, 197, 1), (197, 1, 1), 0); del buf537  # reuse
        buf540 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4, l__mod___post_network_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_41.run(buf539, buf510, buf526, primals_227, buf534, primals_233, primals_234, primals_235, buf535, buf536, buf540, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_227
        del primals_233
        del primals_235
        buf541 = buf516; del buf516  # reuse
        # Source Nodes: [l__mod___post_network_1_attn_kv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (1576, 384), (384, 1), 0), reinterpret_tensor(primals_236, (384, 768), (1, 384), 0), out=buf541)
        buf542 = buf534; del buf534  # reuse
        # Source Nodes: [l__mod___post_network_1_attn_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (8, 384), (75648, 1), 0), reinterpret_tensor(primals_237, (384, 384), (1, 384), 0), out=buf542)
        buf543 = reinterpret_tensor(buf542, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf542  # reuse
        # Source Nodes: [mul_19], Original ATen: [aten.mul]
        triton_poi_fused_mul_35.run(buf543, 3072, grid=grid(3072), stream=stream0)
        buf544 = empty((8, 12, 32, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_65], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf541, buf544, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf545 = buf520; del buf520  # reuse
        # Source Nodes: [attn_65], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf543, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf544, (96, 32, 197), (6304, 197, 1), 0), out=buf545)
        buf548 = empty((8, 12, 1, 197), device='cuda', dtype=torch.float32)
        buf574 = empty((8, 12, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_66, attn_67], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_37.run(buf545, buf548, buf574, 96, 197, grid=grid(96), stream=stream0)
        del buf545
        buf549 = empty((8, 12, 197, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf541, buf549, 605184, grid=grid(605184), stream=stream0)
        del buf541
        buf550 = reinterpret_tensor(buf526, (96, 1, 32), (32, 32, 1), 0); del buf526  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf548, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf549, (96, 197, 32), (6304, 32, 1), 0), out=buf550)
        buf551 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf550, (8, 384), (384, 1), 0), reinterpret_tensor(primals_238, (384, 384), (1, 384), 0), out=buf551)
        buf555 = empty((8, 1, 384), device='cuda', dtype=torch.float32)
        buf556 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf573 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_embed_10, l__mod___post_network_1_norm2, x_225], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_39.run(buf535, buf551, primals_239, primals_240, primals_241, buf555, buf556, buf573, 8, 384, grid=grid(8), stream=stream0)
        del primals_241
        buf557 = empty((8, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_243, buf556, reinterpret_tensor(primals_242, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf557)
        del primals_243
        buf558 = empty((8, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226, x_229], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf557, buf558, 9216, grid=grid(9216), stream=stream0)
        buf559 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf558, reinterpret_tensor(primals_244, (1152, 384), (1, 1152), 0), out=buf559)
        buf560 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf561 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf562 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf564 = reinterpret_tensor(buf562, (8, 197, 1), (197, 1, 1), 0); del buf562  # reuse
        buf565 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3, x_234], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_41.run(buf564, buf535, buf551, primals_239, buf559, primals_245, primals_246, primals_247, buf560, buf561, buf565, 1576, 384, grid=grid(1576), stream=stream0)
        del buf551
        del buf559
        del primals_239
        del primals_245
        del primals_247
        buf566 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf565, (8, 384), (75648, 1), 0), reinterpret_tensor(primals_248, (384, 1000), (1, 384), 0), out=buf566)
        buf567 = buf509; del buf509  # reuse
        # Source Nodes: [aux], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_42.run(buf565, buf567, 602112, grid=grid(602112), stream=stream0)
        buf568 = empty((1568, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [aux], Original ATen: [aten.mm]
        extern_kernels.mm(buf567, reinterpret_tensor(primals_250, (384, 1000), (1, 384), 0), out=buf568)
        buf569 = empty_strided((8, 1000, 2), (2000, 1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [aux, max_1], Original ATen: [aten.add, aten.max]
        triton_red_fused_add_max_43.run(buf568, primals_251, buf569, 16000, 98, grid=grid(16000), stream=stream0)
        buf572 = buf566; del buf566  # reuse
        # Source Nodes: [aux, max_1, mul_20, pred], Original ATen: [aten.add, aten.max, aten.mul]
        triton_per_fused_add_max_mul_44.run(buf572, buf569, primals_249, 8000, 2, grid=grid(8000), stream=stream0)
        del buf569
        del primals_249
        buf571 = empty((8, 1000), device='cuda', dtype=torch.int64)
        # Source Nodes: [aux, max_1], Original ATen: [aten.add, aten.max]
        triton_red_fused_add_max_45.run(buf568, primals_251, buf571, 8000, 196, grid=grid(8000), stream=stream0)
        del buf568
        del primals_251
        # Source Nodes: [l__mod___patch_embed_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(primals_254, primals_254, 1, grid=grid(1), stream=stream0)
        del primals_254
        # Source Nodes: [l__mod___patch_embed_conv_4], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(primals_257, primals_257, 1, grid=grid(1), stream=stream0)
        del primals_257
        # Source Nodes: [l__mod___patch_embed_conv_7], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(primals_260, primals_260, 1, grid=grid(1), stream=stream0)
        del primals_260
        return (buf572, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_14, primals_21, primals_27, primals_34, primals_40, primals_47, primals_53, primals_60, primals_66, primals_68, primals_73, primals_79, primals_84, primals_90, primals_95, primals_101, primals_106, primals_112, primals_117, primals_123, primals_128, primals_134, primals_139, primals_145, primals_150, primals_156, primals_161, primals_167, primals_172, primals_178, primals_183, primals_189, primals_194, primals_200, primals_205, primals_211, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_261, buf0, buf7, buf8, buf9, buf16, buf17, buf18, buf25, buf26, buf34, buf35, buf37, reinterpret_tensor(buf37, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf38, buf39, buf46, buf52, buf60, buf61, buf62, buf63, buf69, buf70, buf72, buf73, buf85, buf90, buf91, buf92, buf93, buf99, buf100, buf102, buf103, buf115, buf120, buf121, buf122, buf123, buf129, buf130, buf132, buf133, buf145, buf150, buf151, buf152, buf153, buf155, buf163, buf164, buf174, buf180, buf181, buf182, buf183, buf188, buf189, buf199, buf205, buf206, buf207, buf208, buf213, buf214, buf224, buf230, buf231, buf232, buf233, buf238, buf239, buf249, buf255, buf256, buf257, buf258, buf263, buf264, buf274, buf280, buf281, buf282, buf283, buf288, buf289, buf299, buf305, buf306, buf307, buf308, buf313, buf314, buf324, buf330, buf331, buf332, buf333, buf338, buf339, buf349, buf355, buf356, buf357, buf358, buf363, buf364, buf374, buf380, buf381, buf382, buf383, buf388, buf389, buf399, buf405, buf406, buf407, buf408, buf413, buf414, buf424, buf430, buf431, buf432, buf433, buf438, buf439, buf449, buf455, buf456, buf457, buf458, buf463, buf464, buf474, buf480, buf481, buf482, buf483, buf488, buf489, buf499, buf505, buf506, buf507, buf508, buf510, buf511, buf514, reinterpret_tensor(buf515, (1576, 384), (384, 1), 0), reinterpret_tensor(buf515, (8, 384), (75648, 1), 0), reinterpret_tensor(buf525, (8, 384), (384, 1), 0), buf530, buf531, buf532, buf533, buf535, buf536, buf539, reinterpret_tensor(buf540, (1576, 384), (384, 1), 0), reinterpret_tensor(buf540, (8, 384), (75648, 1), 0), reinterpret_tensor(buf550, (8, 384), (384, 1), 0), buf555, buf556, buf557, buf558, buf560, buf561, buf564, reinterpret_tensor(buf565, (8, 384), (75648, 1), 0), buf567, reinterpret_tensor(buf571, (8, 1, 1000), (1000, 1000, 1), 0), reinterpret_tensor(primals_250, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_248, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_244, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_242, (1152, 384), (384, 1), 0), buf573, reinterpret_tensor(primals_238, (384, 384), (384, 1), 0), reinterpret_tensor(buf548, (96, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf549, (96, 32, 197), (6304, 1, 32), 0), buf574, reinterpret_tensor(buf543, (96, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf544, (96, 197, 32), (6304, 1, 197), 0), reinterpret_tensor(primals_237, (384, 384), (384, 1), 0), reinterpret_tensor(primals_236, (768, 384), (384, 1), 0), reinterpret_tensor(primals_232, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_230, (1152, 384), (384, 1), 0), buf575, reinterpret_tensor(primals_226, (384, 384), (384, 1), 0), reinterpret_tensor(buf523, (96, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf524, (96, 32, 197), (6304, 1, 32), 0), buf576, reinterpret_tensor(buf518, (96, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf519, (96, 197, 32), (6304, 1, 197), 0), reinterpret_tensor(primals_225, (384, 384), (384, 1), 0), reinterpret_tensor(primals_224, (768, 384), (384, 1), 0), reinterpret_tensor(primals_220, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_218, (1152, 384), (384, 1), 0), buf577, reinterpret_tensor(primals_214, (384, 384), (384, 1), 0), reinterpret_tensor(buf496, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf497, (96, 32, 196), (6272, 1, 32), 0), buf578, reinterpret_tensor(buf491, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf492, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_213, (1152, 384), (384, 1), 0), buf579, reinterpret_tensor(primals_209, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_207, (1152, 384), (384, 1), 0), buf580, reinterpret_tensor(primals_203, (384, 384), (384, 1), 0), reinterpret_tensor(buf471, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf472, (96, 32, 196), (6272, 1, 32), 0), buf581, reinterpret_tensor(buf466, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf467, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_202, (1152, 384), (384, 1), 0), buf582, reinterpret_tensor(primals_198, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_196, (1152, 384), (384, 1), 0), buf583, reinterpret_tensor(primals_192, (384, 384), (384, 1), 0), reinterpret_tensor(buf446, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf447, (96, 32, 196), (6272, 1, 32), 0), buf584, reinterpret_tensor(buf441, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf442, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_191, (1152, 384), (384, 1), 0), buf585, reinterpret_tensor(primals_187, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_185, (1152, 384), (384, 1), 0), buf586, reinterpret_tensor(primals_181, (384, 384), (384, 1), 0), reinterpret_tensor(buf421, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf422, (96, 32, 196), (6272, 1, 32), 0), buf587, reinterpret_tensor(buf416, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf417, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_180, (1152, 384), (384, 1), 0), buf588, reinterpret_tensor(primals_176, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_174, (1152, 384), (384, 1), 0), buf589, reinterpret_tensor(primals_170, (384, 384), (384, 1), 0), reinterpret_tensor(buf396, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf397, (96, 32, 196), (6272, 1, 32), 0), buf590, reinterpret_tensor(buf391, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf392, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_169, (1152, 384), (384, 1), 0), buf591, reinterpret_tensor(primals_165, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_163, (1152, 384), (384, 1), 0), buf592, reinterpret_tensor(primals_159, (384, 384), (384, 1), 0), reinterpret_tensor(buf371, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf372, (96, 32, 196), (6272, 1, 32), 0), buf593, reinterpret_tensor(buf366, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf367, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_158, (1152, 384), (384, 1), 0), buf594, reinterpret_tensor(primals_154, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_152, (1152, 384), (384, 1), 0), buf595, reinterpret_tensor(primals_148, (384, 384), (384, 1), 0), reinterpret_tensor(buf346, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf347, (96, 32, 196), (6272, 1, 32), 0), buf596, reinterpret_tensor(buf341, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf342, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_147, (1152, 384), (384, 1), 0), buf597, reinterpret_tensor(primals_143, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_141, (1152, 384), (384, 1), 0), buf598, reinterpret_tensor(primals_137, (384, 384), (384, 1), 0), reinterpret_tensor(buf321, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf322, (96, 32, 196), (6272, 1, 32), 0), buf599, reinterpret_tensor(buf316, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf317, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_136, (1152, 384), (384, 1), 0), buf600, reinterpret_tensor(primals_132, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_130, (1152, 384), (384, 1), 0), buf601, reinterpret_tensor(primals_126, (384, 384), (384, 1), 0), reinterpret_tensor(buf296, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf297, (96, 32, 196), (6272, 1, 32), 0), buf602, reinterpret_tensor(buf291, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf292, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_125, (1152, 384), (384, 1), 0), buf603, reinterpret_tensor(primals_121, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_119, (1152, 384), (384, 1), 0), buf604, reinterpret_tensor(primals_115, (384, 384), (384, 1), 0), reinterpret_tensor(buf271, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf272, (96, 32, 196), (6272, 1, 32), 0), buf605, reinterpret_tensor(buf266, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf267, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_114, (1152, 384), (384, 1), 0), buf606, reinterpret_tensor(primals_110, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_108, (1152, 384), (384, 1), 0), buf607, reinterpret_tensor(primals_104, (384, 384), (384, 1), 0), reinterpret_tensor(buf246, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf247, (96, 32, 196), (6272, 1, 32), 0), buf608, reinterpret_tensor(buf241, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf242, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_103, (1152, 384), (384, 1), 0), buf609, reinterpret_tensor(primals_99, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_97, (1152, 384), (384, 1), 0), buf610, reinterpret_tensor(primals_93, (384, 384), (384, 1), 0), reinterpret_tensor(buf221, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf222, (96, 32, 196), (6272, 1, 32), 0), buf611, reinterpret_tensor(buf216, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf217, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_92, (1152, 384), (384, 1), 0), buf612, reinterpret_tensor(primals_88, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_86, (1152, 384), (384, 1), 0), buf613, reinterpret_tensor(primals_82, (384, 384), (384, 1), 0), reinterpret_tensor(buf196, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf197, (96, 32, 196), (6272, 1, 32), 0), buf614, reinterpret_tensor(buf191, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf192, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_81, (1152, 384), (384, 1), 0), buf615, reinterpret_tensor(primals_77, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_75, (1152, 384), (384, 1), 0), buf616, reinterpret_tensor(primals_71, (384, 384), (384, 1), 0), reinterpret_tensor(buf171, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf172, (96, 32, 196), (6272, 1, 32), 0), buf617, reinterpret_tensor(buf166, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf167, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_70, (1152, 384), (384, 1), 0), buf618, reinterpret_tensor(primals_64, (192, 576), (576, 1), 0), reinterpret_tensor(primals_62, (576, 192), (192, 1), 0), buf619, reinterpret_tensor(primals_58, (192, 192), (192, 1), 0), reinterpret_tensor(buf137, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf138, (9408, 32, 9), (288, 1, 32), 0), buf620, reinterpret_tensor(primals_56, (486, 192), (192, 1), 0), reinterpret_tensor(primals_55, (192, 192), (192, 1), 0), buf621, reinterpret_tensor(primals_51, (192, 576), (576, 1), 0), reinterpret_tensor(primals_49, (576, 192), (192, 1), 0), buf622, reinterpret_tensor(primals_45, (192, 192), (192, 1), 0), reinterpret_tensor(buf107, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf108, (9408, 32, 9), (288, 1, 32), 0), buf623, reinterpret_tensor(primals_43, (486, 192), (192, 1), 0), reinterpret_tensor(primals_42, (192, 192), (192, 1), 0), buf624, reinterpret_tensor(primals_38, (192, 576), (576, 1), 0), reinterpret_tensor(primals_36, (576, 192), (192, 1), 0), buf625, reinterpret_tensor(primals_32, (192, 192), (192, 1), 0), reinterpret_tensor(buf77, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf78, (9408, 32, 9), (288, 1, 32), 0), buf626, reinterpret_tensor(primals_30, (486, 192), (192, 1), 0), reinterpret_tensor(primals_29, (192, 192), (192, 1), 0), buf627, reinterpret_tensor(primals_25, (192, 576), (576, 1), 0), reinterpret_tensor(primals_23, (576, 192), (192, 1), 0), buf628, reinterpret_tensor(primals_19, (192, 192), (192, 1), 0), reinterpret_tensor(buf43, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf44, (9408, 32, 9), (288, 1, 32), 0), buf629, reinterpret_tensor(primals_17, (486, 192), (192, 1), 0), reinterpret_tensor(primals_16, (192, 192), (192, 1), 0), buf630, reinterpret_tensor(buf22, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf13, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((192, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('volo_d1_224', benchmark_compiled_module)
