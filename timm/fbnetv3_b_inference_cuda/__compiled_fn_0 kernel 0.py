
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


# kernel path: /tmp/torchinductor_youkaichao/rr/crrwkxcpboofanir4ahdijl45q63dz5oiulxumeqxm5cu3ocy5kg.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/zy/czyirduix6cyyuec7dfvcmlnzfoyrtbb4zm5cpjn54ggbwkfp337.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaojitkkgqxk3s7pcih5u4tkvxncpfxhjdi4tw2xhlyxhtnxkeb.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# shortcut => add_2, clamp_max, clamp_min, div, mul_3
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (16*x2) + (262144*y1)), tmp22, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccd2jzwp6hv3mr3jcaa2jm7ux2txehudfnaok4ejqszntlpc4b7q.py
# Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_1 => add_8
# x_12 => add_7, mul_10, mul_9, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_add_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16384
    y1 = (yindex // 16384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16384*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (16*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/cszwwzpvo57nfobnraqhkagc57rwbgi37awmtsvj2z56pa63g2dd.py
# Source Nodes: [x_30, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_30 => add_16, mul_19, mul_20, sub_5
# x_33 => add_17, clamp_max_3, clamp_min_3, div_3, mul_21
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp22, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbbfths2hqgjuspq2yupwxdyu7kmsst7k6i7krop2opxlecvgjg.py
# Source Nodes: [x_35, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_35 => add_19, mul_23, mul_24, sub_6
# x_38 => add_20, clamp_max_4, clamp_min_4, div_4, mul_25
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp22, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3lthoosu65ao2bwebgee2lezpyux4c5eqbtuindwzpytetg2wk.py
# Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_41 => add_22, mul_27, mul_28, sub_7
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (24*x2) + (98304*y1)), tmp14, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vy/cvybpomqkddc3qgrnverlqcuef3ghrq4ncnakukqr4obdz54sm3c.py
# Source Nodes: [x_46, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_46 => add_24, mul_30, mul_31, sub_8
# x_49 => add_25, clamp_max_5, clamp_min_5, div_5, mul_32
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (48*x2) + (196608*y1)), tmp22, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxr7kobtxbsrjvb3eoyaibh7tch2mdni6pkzecxxdeyl7cywuwxw.py
# Source Nodes: [shortcut_4, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_4 => add_31
# x_57 => add_30, mul_38, mul_39, sub_10
triton_poi_fused__native_batch_norm_legit_no_training_add_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 24
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (4096*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (24*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2ti7abhiusv3eygtjdvpedqh4poxll73noyxl7ierq2rxmittb.py
# Source Nodes: [x_100, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_100 => add_52, clamp_max_11, clamp_min_11, div_11, mul_65
# x_97 => add_51, mul_63, mul_64, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (120*x2) + (491520*y1)), tmp22, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5a7bwscteuh3rb4ifxfjo4byisnmg52qhtjhinm7rwv2a6jdud.py
# Source Nodes: [x_102, x_105, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_102 => add_54, mul_67, mul_68, sub_18
# x_105 => add_55, clamp_max_12, clamp_min_12, div_12, mul_69
# x_se => mean
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 960
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = 1024.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgtkzuw4l37gfpz2ff4pwa7vxhavtttenx77iatuxxfauycswy7.py
# Source Nodes: [x_105, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
# x_105 => add_55, clamp_max_12, clamp_min_12, div_12, mul_69
# x_se => mean
# x_se_1 => convolution_19
# x_se_2 => add_56, clamp_max_13, clamp_min_13, div_13, mul_70
triton_poi_fused_convolution_hardswish_mean_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2dkjxscep5uucepqvwy3bbul265oqmpk6v5j6gqnpyb6na6yjz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_105, x_106, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => add_57, clamp_max_14, clamp_min_14, div_14
# x_105 => add_55, clamp_max_12, clamp_min_12, div_12, mul_69
# x_106 => mul_71
# x_se => mean
# x_se_1 => convolution_19
# x_se_2 => add_56, clamp_max_13, clamp_min_13, div_13, mul_70
# x_se_3 => convolution_20
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (120*x2) + (122880*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5t3wuakzcqow5okzor6cgiszfi5xmmxrbq3keoydxzhw4tji5f.py
# Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_108 => add_59, mul_73, mul_74, sub_19
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (40*x2) + (40960*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrp5aix7jry5q4xkx3f3megvbhm4sui6ahj6v4sd5om4evslm2m.py
# Source Nodes: [x_113, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_113 => add_61, mul_76, mul_77, sub_20
# x_116 => add_62, clamp_max_15, clamp_min_15, div_15, mul_78
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (120*x2) + (122880*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxp6gqnltwodn7eb4z53zjpweqfpckzdzp4n4iwbfqlze55a62o.py
# Source Nodes: [x_121, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
# x_121 => add_65, clamp_max_16, clamp_min_16, div_16, mul_82
# x_se_4 => mean_1
# x_se_5 => convolution_24
# x_se_6 => add_66, clamp_max_17, clamp_min_17, div_17, mul_83
triton_poi_fused_convolution_hardswish_mean_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkwtpasz4cmfm7rtbs3ntzdry3p5chjciimjs46dpzp76acfahb.py
# Source Nodes: [shortcut_8, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_8 => add_70
# x_124 => add_69, mul_86, mul_87, sub_22
triton_poi_fused__native_batch_norm_legit_no_training_add_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 40
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (40*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (40*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/me/cmexigixxm2uc2rqkb2gyrok226shik4rv45m3qdbdapetxud3ms.py
# Source Nodes: [x_181, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_181 => add_105, mul_128, mul_129, sub_32
# x_184 => add_106, clamp_max_31, clamp_min_31, div_31, mul_130
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1600
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 200
    y1 = (yindex // 200)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (200*x2) + (204800*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvgz2naeniyjzfigfdujfjgwxrhs6lpm2owpmqdygwzdgbknylh7.py
# Source Nodes: [x_186, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_186 => add_108, mul_132, mul_133, sub_33
# x_189 => add_109, clamp_max_32, clamp_min_32, div_32, mul_134
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1600
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 200
    y1 = (yindex // 200)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (200*x2) + (51200*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxgd55wv5xrpxeocggxdc22tigqsdqjonuhrsxb4raxkmzfxegm.py
# Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_192 => add_111, mul_136, mul_137, sub_34
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (72*x2) + (18432*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crdc2f6hnmxi75i3qgwzh2rs2sz4f6w7xmszh7y5vwytst3igxai.py
# Source Nodes: [x_197, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_197 => add_113, mul_139, mul_140, sub_35
# x_200 => add_114, clamp_max_33, clamp_min_33, div_33, mul_141
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 216
    y1 = (yindex // 216)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (216*x2) + (55296*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czmube63r7dtxtx5gphoccb2nat3lsbicxuebkiola2ahktpocx2.py
# Source Nodes: [shortcut_13, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_13 => add_120
# x_208 => add_119, mul_147, mul_148, sub_37
triton_poi_fused__native_batch_norm_legit_no_training_add_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 72
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (18432*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (72*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (72*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckd7r6kqbjgjrdvehzdvlxgp6fkipiq4vr2eayry3vsiqqcru7ei.py
# Source Nodes: [x_265, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_265 => add_149, mul_183, mul_184, sub_47
# x_268 => add_150, clamp_max_41, clamp_min_41, div_41, mul_185
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2880
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 360
    y1 = (yindex // 360)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (360*x2) + (92160*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgktd5aqvxzdcjlq65ta2noviuqrysp44kk272i6dqhzars6xtc.py
# Source Nodes: [x_270, x_273, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_270 => add_152, mul_187, mul_188, sub_48
# x_273 => add_153, clamp_max_42, clamp_min_42, div_42, mul_189
# x_se_20 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 2880
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 360
    tmp0 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = 256.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxpqv2tlkhennzu2k22plvt4qm7cbhy7gpenmqsxu6kya44xr4o.py
# Source Nodes: [x_273, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
# x_273 => add_153, clamp_max_42, clamp_min_42, div_42, mul_189
# x_se_20 => mean_5
# x_se_21 => convolution_59
# x_se_22 => add_154, clamp_max_43, clamp_min_43, div_43, mul_190
triton_poi_fused_convolution_hardswish_mean_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7qbi76dbrfnci73e2zidvkijsbjyodfxebujsgticrjnu6dtt2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_273, x_274, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => add_155, clamp_max_44, clamp_min_44, div_44
# x_273 => add_153, clamp_max_42, clamp_min_42, div_42, mul_189
# x_274 => mul_191
# x_se_20 => mean_5
# x_se_21 => convolution_59
# x_se_22 => add_154, clamp_max_43, clamp_min_43, div_43, mul_190
# x_se_23 => convolution_60
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2880
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 360
    y1 = (yindex // 360)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (360*x2) + (92160*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltyng3y65cljansbwjrigkjlob5tet6pv3rijy2t6en3snx5k3o.py
# Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_276 => add_157, mul_193, mul_194, sub_49
triton_poi_fused__native_batch_norm_legit_no_training_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (120*x2) + (30720*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/clerdjegu7hhy2wdvgvica37y24uuh6tg5fgmllf3cnrrbusbcy2.py
# Source Nodes: [x_289, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
# x_289 => add_163, clamp_max_46, clamp_min_46, div_46, mul_202
# x_se_24 => mean_6
# x_se_25 => convolution_64
# x_se_26 => add_164, clamp_max_47, clamp_min_47, div_47, mul_203
triton_poi_fused_convolution_hardswish_mean_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hdvgz47d255gkqo7uh2qmdockgbuqnb2vjnls7war6cuo7dg2c.py
# Source Nodes: [shortcut_18, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_18 => add_168
# x_292 => add_167, mul_206, mul_207, sub_52
triton_poi_fused__native_batch_norm_legit_no_training_add_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 120
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (30720*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (120*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (120*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6q65u5vmkibfcdb7cujq3hze3tn2327kg4fr42hxrwptorwxxu.py
# Source Nodes: [x_366, x_369], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_366 => add_214, mul_261, mul_262, sub_65
# x_369 => add_215, clamp_max_65, clamp_min_65, div_65, mul_263
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5760
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 720
    y1 = (yindex // 720)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (720*x2) + (184320*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdxfwmlkxgae7uekxbhzvanrw2bikcctbalsybp2rgdxrmv6bks.py
# Source Nodes: [x_371, x_374, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_371 => add_217, mul_265, mul_266, sub_66
# x_374 => add_218, clamp_max_66, clamp_min_66, div_66, mul_267
# x_se_44 => mean_11
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 720
    tmp0 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 64.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kz/ckzmf6cm7ndorlbx35lfya3lrgcpkz7nqblyku6fchjwhwvwuq2n.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_374, x_375, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => add_220, clamp_max_68, clamp_min_68, div_68
# x_374 => add_218, clamp_max_66, clamp_min_66, div_66, mul_267
# x_375 => mul_269
# x_se_44 => mean_11
# x_se_45 => convolution_89
# x_se_46 => add_219, clamp_max_67, clamp_min_67, div_67, mul_268
# x_se_47 => convolution_90
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5760
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 720
    y1 = (yindex // 720)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (720*x2) + (46080*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zb/czbstweqxyqrw35vpziyxmk3jxl3tcdek2zlnczp6cnsiypzefng.py
# Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_377 => add_222, mul_271, mul_272, sub_67
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (184*x2) + (11776*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijpbflb6zupe7evif53ppyfq6tcafwpzcxom5icj4bvdmsjvofr.py
# Source Nodes: [x_382, x_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_382 => add_224, mul_274, mul_275, sub_68
# x_385 => add_225, clamp_max_69, clamp_min_69, div_69, mul_276
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5888
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 736
    y1 = (yindex // 736)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (736*x2) + (47104*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmauwu7pcgqw44z4pflf2fayy577zrxdyysitopuwkvpev2k3hzb.py
# Source Nodes: [x_387, x_390, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_387 => add_227, mul_278, mul_279, sub_69
# x_390 => add_228, clamp_max_70, clamp_min_70, div_70, mul_280
# x_se_48 => mean_12
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_34', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 736
    tmp0 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 64.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ye/cyejjxixx76r3f3mdejrlcamv4u2spnospl4omzb6lketenogybl.py
# Source Nodes: [x_390, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
# x_390 => add_228, clamp_max_70, clamp_min_70, div_70, mul_280
# x_se_48 => mean_12
# x_se_49 => convolution_94
# x_se_50 => add_229, clamp_max_71, clamp_min_71, div_71, mul_281
triton_poi_fused_convolution_hardswish_mean_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/ceg6lzsd67uomrwxepknsf6ylutv4r3fujhxmi2uihszur4j5ogg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_390, x_391, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => add_230, clamp_max_72, clamp_min_72, div_72
# x_390 => add_228, clamp_max_70, clamp_min_70, div_70, mul_280
# x_391 => mul_282
# x_se_48 => mean_12
# x_se_49 => convolution_94
# x_se_50 => add_229, clamp_max_71, clamp_min_71, div_71, mul_281
# x_se_51 => convolution_95
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5888
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 736
    y1 = (yindex // 736)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (736*x2) + (47104*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqtusmgl3k2nla74jrz6cv7ean5ulkps3sxc7divonlhhfvm6lr.py
# Source Nodes: [shortcut_24, x_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_24 => add_233
# x_393 => add_232, mul_284, mul_285, sub_70
triton_poi_fused__native_batch_norm_legit_no_training_add_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 184
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (11776*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (184*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (184*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6kzejofnpeaqps73z3hxdrsxdfvyagsmg5vshynuz2fu5wt4ac.py
# Source Nodes: [x_467, x_470], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_467 => add_279, mul_339, mul_340, sub_83
# x_470 => add_280, clamp_max_89, clamp_min_89, div_89, mul_341
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8832
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1104
    y1 = (yindex // 1104)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (1104*x2) + (70656*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coee2cheo6ulzu3jkbxdxgbs7vd32urd64z4gb3lcmdthorueflt.py
# Source Nodes: [x_472, x_475, x_se_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_472 => add_282, mul_343, mul_344, sub_84
# x_475 => add_283, clamp_max_90, clamp_min_90, div_90, mul_345
# x_se_68 => mean_17
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8832
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1104
    tmp0 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 64.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshqkf5uitlpygqagfegs37rzivaeqkxsbiq6fdpzmc2cstza3d6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____6___se_gate, x_475, x_476, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
# getattr_getattr_l__mod___blocks___5_____6___se_gate => add_285, clamp_max_92, clamp_min_92, div_92
# x_475 => add_283, clamp_max_90, clamp_min_90, div_90, mul_345
# x_476 => mul_347
# x_se_68 => mean_17
# x_se_69 => convolution_119
# x_se_70 => add_284, clamp_max_91, clamp_min_91, div_91, mul_346
# x_se_71 => convolution_120
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8832
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1104
    y1 = (yindex // 1104)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (1104*x2) + (70656*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wu/cwunjb635bxzeof7bwsriv2upgya4bhx4mpgqbqzjbqwrmaqqpcw.py
# Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_478 => add_287, mul_349, mul_350, sub_85
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (224*x2) + (14336*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cvifm3pzdtp334eb6eap3qjfb7me4zsusum7t2j4ag2dln5ozyqx.py
# Source Nodes: [x_483, x_488, x_489], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_483 => add_289, mul_352, mul_353, sub_86
# x_488 => add_290, clamp_max_93, clamp_min_93, div_93, mul_354
# x_489 => mean_18
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_42', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1344
    tmp0 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 64.0
    tmp28 = tmp26 / tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvmfggwytktfgdy3ohafubsqaqbbjyhvoamtvyp5tut243g7xhp.py
# Source Nodes: [x_493], Original ATen: [aten.hardswish]
# x_493 => add_291, clamp_max_94, clamp_min_94, div_94, mul_355
triton_poi_fused_hardswish_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, ), (1, ))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (24, ), (1, ))
    assert_size_stride(arg15_1, (24, ), (1, ))
    assert_size_stride(arg16_1, (48, ), (1, ))
    assert_size_stride(arg17_1, (48, ), (1, ))
    assert_size_stride(arg18_1, (48, ), (1, ))
    assert_size_stride(arg19_1, (48, ), (1, ))
    assert_size_stride(arg20_1, (24, ), (1, ))
    assert_size_stride(arg21_1, (24, ), (1, ))
    assert_size_stride(arg22_1, (48, ), (1, ))
    assert_size_stride(arg23_1, (48, ), (1, ))
    assert_size_stride(arg24_1, (48, ), (1, ))
    assert_size_stride(arg25_1, (48, ), (1, ))
    assert_size_stride(arg26_1, (24, ), (1, ))
    assert_size_stride(arg27_1, (24, ), (1, ))
    assert_size_stride(arg28_1, (48, ), (1, ))
    assert_size_stride(arg29_1, (48, ), (1, ))
    assert_size_stride(arg30_1, (48, ), (1, ))
    assert_size_stride(arg31_1, (48, ), (1, ))
    assert_size_stride(arg32_1, (24, ), (1, ))
    assert_size_stride(arg33_1, (24, ), (1, ))
    assert_size_stride(arg34_1, (120, ), (1, ))
    assert_size_stride(arg35_1, (120, ), (1, ))
    assert_size_stride(arg36_1, (120, ), (1, ))
    assert_size_stride(arg37_1, (120, ), (1, ))
    assert_size_stride(arg38_1, (40, ), (1, ))
    assert_size_stride(arg39_1, (40, ), (1, ))
    assert_size_stride(arg40_1, (120, ), (1, ))
    assert_size_stride(arg41_1, (120, ), (1, ))
    assert_size_stride(arg42_1, (120, ), (1, ))
    assert_size_stride(arg43_1, (120, ), (1, ))
    assert_size_stride(arg44_1, (40, ), (1, ))
    assert_size_stride(arg45_1, (40, ), (1, ))
    assert_size_stride(arg46_1, (120, ), (1, ))
    assert_size_stride(arg47_1, (120, ), (1, ))
    assert_size_stride(arg48_1, (120, ), (1, ))
    assert_size_stride(arg49_1, (120, ), (1, ))
    assert_size_stride(arg50_1, (40, ), (1, ))
    assert_size_stride(arg51_1, (40, ), (1, ))
    assert_size_stride(arg52_1, (120, ), (1, ))
    assert_size_stride(arg53_1, (120, ), (1, ))
    assert_size_stride(arg54_1, (120, ), (1, ))
    assert_size_stride(arg55_1, (120, ), (1, ))
    assert_size_stride(arg56_1, (40, ), (1, ))
    assert_size_stride(arg57_1, (40, ), (1, ))
    assert_size_stride(arg58_1, (120, ), (1, ))
    assert_size_stride(arg59_1, (120, ), (1, ))
    assert_size_stride(arg60_1, (120, ), (1, ))
    assert_size_stride(arg61_1, (120, ), (1, ))
    assert_size_stride(arg62_1, (40, ), (1, ))
    assert_size_stride(arg63_1, (40, ), (1, ))
    assert_size_stride(arg64_1, (200, ), (1, ))
    assert_size_stride(arg65_1, (200, ), (1, ))
    assert_size_stride(arg66_1, (200, ), (1, ))
    assert_size_stride(arg67_1, (200, ), (1, ))
    assert_size_stride(arg68_1, (72, ), (1, ))
    assert_size_stride(arg69_1, (72, ), (1, ))
    assert_size_stride(arg70_1, (216, ), (1, ))
    assert_size_stride(arg71_1, (216, ), (1, ))
    assert_size_stride(arg72_1, (216, ), (1, ))
    assert_size_stride(arg73_1, (216, ), (1, ))
    assert_size_stride(arg74_1, (72, ), (1, ))
    assert_size_stride(arg75_1, (72, ), (1, ))
    assert_size_stride(arg76_1, (216, ), (1, ))
    assert_size_stride(arg77_1, (216, ), (1, ))
    assert_size_stride(arg78_1, (216, ), (1, ))
    assert_size_stride(arg79_1, (216, ), (1, ))
    assert_size_stride(arg80_1, (72, ), (1, ))
    assert_size_stride(arg81_1, (72, ), (1, ))
    assert_size_stride(arg82_1, (216, ), (1, ))
    assert_size_stride(arg83_1, (216, ), (1, ))
    assert_size_stride(arg84_1, (216, ), (1, ))
    assert_size_stride(arg85_1, (216, ), (1, ))
    assert_size_stride(arg86_1, (72, ), (1, ))
    assert_size_stride(arg87_1, (72, ), (1, ))
    assert_size_stride(arg88_1, (216, ), (1, ))
    assert_size_stride(arg89_1, (216, ), (1, ))
    assert_size_stride(arg90_1, (216, ), (1, ))
    assert_size_stride(arg91_1, (216, ), (1, ))
    assert_size_stride(arg92_1, (72, ), (1, ))
    assert_size_stride(arg93_1, (72, ), (1, ))
    assert_size_stride(arg94_1, (360, ), (1, ))
    assert_size_stride(arg95_1, (360, ), (1, ))
    assert_size_stride(arg96_1, (360, ), (1, ))
    assert_size_stride(arg97_1, (360, ), (1, ))
    assert_size_stride(arg98_1, (120, ), (1, ))
    assert_size_stride(arg99_1, (120, ), (1, ))
    assert_size_stride(arg100_1, (360, ), (1, ))
    assert_size_stride(arg101_1, (360, ), (1, ))
    assert_size_stride(arg102_1, (360, ), (1, ))
    assert_size_stride(arg103_1, (360, ), (1, ))
    assert_size_stride(arg104_1, (120, ), (1, ))
    assert_size_stride(arg105_1, (120, ), (1, ))
    assert_size_stride(arg106_1, (360, ), (1, ))
    assert_size_stride(arg107_1, (360, ), (1, ))
    assert_size_stride(arg108_1, (360, ), (1, ))
    assert_size_stride(arg109_1, (360, ), (1, ))
    assert_size_stride(arg110_1, (120, ), (1, ))
    assert_size_stride(arg111_1, (120, ), (1, ))
    assert_size_stride(arg112_1, (360, ), (1, ))
    assert_size_stride(arg113_1, (360, ), (1, ))
    assert_size_stride(arg114_1, (360, ), (1, ))
    assert_size_stride(arg115_1, (360, ), (1, ))
    assert_size_stride(arg116_1, (120, ), (1, ))
    assert_size_stride(arg117_1, (120, ), (1, ))
    assert_size_stride(arg118_1, (360, ), (1, ))
    assert_size_stride(arg119_1, (360, ), (1, ))
    assert_size_stride(arg120_1, (360, ), (1, ))
    assert_size_stride(arg121_1, (360, ), (1, ))
    assert_size_stride(arg122_1, (120, ), (1, ))
    assert_size_stride(arg123_1, (120, ), (1, ))
    assert_size_stride(arg124_1, (360, ), (1, ))
    assert_size_stride(arg125_1, (360, ), (1, ))
    assert_size_stride(arg126_1, (360, ), (1, ))
    assert_size_stride(arg127_1, (360, ), (1, ))
    assert_size_stride(arg128_1, (120, ), (1, ))
    assert_size_stride(arg129_1, (120, ), (1, ))
    assert_size_stride(arg130_1, (720, ), (1, ))
    assert_size_stride(arg131_1, (720, ), (1, ))
    assert_size_stride(arg132_1, (720, ), (1, ))
    assert_size_stride(arg133_1, (720, ), (1, ))
    assert_size_stride(arg134_1, (184, ), (1, ))
    assert_size_stride(arg135_1, (184, ), (1, ))
    assert_size_stride(arg136_1, (736, ), (1, ))
    assert_size_stride(arg137_1, (736, ), (1, ))
    assert_size_stride(arg138_1, (736, ), (1, ))
    assert_size_stride(arg139_1, (736, ), (1, ))
    assert_size_stride(arg140_1, (184, ), (1, ))
    assert_size_stride(arg141_1, (184, ), (1, ))
    assert_size_stride(arg142_1, (736, ), (1, ))
    assert_size_stride(arg143_1, (736, ), (1, ))
    assert_size_stride(arg144_1, (736, ), (1, ))
    assert_size_stride(arg145_1, (736, ), (1, ))
    assert_size_stride(arg146_1, (184, ), (1, ))
    assert_size_stride(arg147_1, (184, ), (1, ))
    assert_size_stride(arg148_1, (736, ), (1, ))
    assert_size_stride(arg149_1, (736, ), (1, ))
    assert_size_stride(arg150_1, (736, ), (1, ))
    assert_size_stride(arg151_1, (736, ), (1, ))
    assert_size_stride(arg152_1, (184, ), (1, ))
    assert_size_stride(arg153_1, (184, ), (1, ))
    assert_size_stride(arg154_1, (736, ), (1, ))
    assert_size_stride(arg155_1, (736, ), (1, ))
    assert_size_stride(arg156_1, (736, ), (1, ))
    assert_size_stride(arg157_1, (736, ), (1, ))
    assert_size_stride(arg158_1, (184, ), (1, ))
    assert_size_stride(arg159_1, (184, ), (1, ))
    assert_size_stride(arg160_1, (736, ), (1, ))
    assert_size_stride(arg161_1, (736, ), (1, ))
    assert_size_stride(arg162_1, (736, ), (1, ))
    assert_size_stride(arg163_1, (736, ), (1, ))
    assert_size_stride(arg164_1, (184, ), (1, ))
    assert_size_stride(arg165_1, (184, ), (1, ))
    assert_size_stride(arg166_1, (1104, ), (1, ))
    assert_size_stride(arg167_1, (1104, ), (1, ))
    assert_size_stride(arg168_1, (1104, ), (1, ))
    assert_size_stride(arg169_1, (1104, ), (1, ))
    assert_size_stride(arg170_1, (224, ), (1, ))
    assert_size_stride(arg171_1, (224, ), (1, ))
    assert_size_stride(arg172_1, (1344, ), (1, ))
    assert_size_stride(arg173_1, (1344, ), (1, ))
    assert_size_stride(arg174_1, (1000, 1984), (1984, 1))
    assert_size_stride(arg175_1, (1000, ), (1, ))
    assert_size_stride(arg176_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg177_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg178_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg179_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg180_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg181_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg182_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg183_1, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg184_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg185_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg186_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg187_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg188_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg189_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg190_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg191_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg192_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg193_1, (120, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg194_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg195_1, (8, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg196_1, (8, ), (1, ))
    assert_size_stride(arg197_1, (120, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg198_1, (120, ), (1, ))
    assert_size_stride(arg199_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg200_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg201_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg202_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg203_1, (16, ), (1, ))
    assert_size_stride(arg204_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg205_1, (120, ), (1, ))
    assert_size_stride(arg206_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg207_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg208_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg209_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg210_1, (16, ), (1, ))
    assert_size_stride(arg211_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg212_1, (120, ), (1, ))
    assert_size_stride(arg213_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg214_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg215_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg216_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg217_1, (16, ), (1, ))
    assert_size_stride(arg218_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg219_1, (120, ), (1, ))
    assert_size_stride(arg220_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg221_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg222_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg223_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg224_1, (16, ), (1, ))
    assert_size_stride(arg225_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg226_1, (120, ), (1, ))
    assert_size_stride(arg227_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg228_1, (200, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg229_1, (200, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg230_1, (72, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg231_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg232_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg233_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg234_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg235_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg236_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg237_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg238_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg239_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg240_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg241_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg242_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg243_1, (360, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg244_1, (360, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg245_1, (24, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg246_1, (24, ), (1, ))
    assert_size_stride(arg247_1, (360, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg248_1, (360, ), (1, ))
    assert_size_stride(arg249_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg250_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg251_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg252_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg253_1, (32, ), (1, ))
    assert_size_stride(arg254_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg255_1, (360, ), (1, ))
    assert_size_stride(arg256_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg257_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg258_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg259_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg260_1, (32, ), (1, ))
    assert_size_stride(arg261_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg262_1, (360, ), (1, ))
    assert_size_stride(arg263_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg264_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg265_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg266_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg267_1, (32, ), (1, ))
    assert_size_stride(arg268_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg269_1, (360, ), (1, ))
    assert_size_stride(arg270_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg271_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg272_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg273_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg274_1, (32, ), (1, ))
    assert_size_stride(arg275_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg276_1, (360, ), (1, ))
    assert_size_stride(arg277_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg278_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg279_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg280_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg281_1, (32, ), (1, ))
    assert_size_stride(arg282_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg283_1, (360, ), (1, ))
    assert_size_stride(arg284_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg285_1, (720, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg286_1, (720, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg287_1, (32, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(arg288_1, (32, ), (1, ))
    assert_size_stride(arg289_1, (720, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg290_1, (720, ), (1, ))
    assert_size_stride(arg291_1, (184, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(arg292_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg293_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg294_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg295_1, (48, ), (1, ))
    assert_size_stride(arg296_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg297_1, (736, ), (1, ))
    assert_size_stride(arg298_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg299_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg300_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg301_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg302_1, (48, ), (1, ))
    assert_size_stride(arg303_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg304_1, (736, ), (1, ))
    assert_size_stride(arg305_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg306_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg307_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg308_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg309_1, (48, ), (1, ))
    assert_size_stride(arg310_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg311_1, (736, ), (1, ))
    assert_size_stride(arg312_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg313_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg314_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg315_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg316_1, (48, ), (1, ))
    assert_size_stride(arg317_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg318_1, (736, ), (1, ))
    assert_size_stride(arg319_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg320_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg321_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg322_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg323_1, (48, ), (1, ))
    assert_size_stride(arg324_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg325_1, (736, ), (1, ))
    assert_size_stride(arg326_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg327_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg328_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg329_1, (48, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg330_1, (48, ), (1, ))
    assert_size_stride(arg331_1, (1104, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg332_1, (1104, ), (1, ))
    assert_size_stride(arg333_1, (224, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg334_1, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg335_1, (1984, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(arg336_1, (16, ), (1, ))
    assert_size_stride(arg337_1, (16, ), (1, ))
    assert_size_stride(arg338_1, (16, ), (1, ))
    assert_size_stride(arg339_1, (16, ), (1, ))
    assert_size_stride(arg340_1, (16, ), (1, ))
    assert_size_stride(arg341_1, (16, ), (1, ))
    assert_size_stride(arg342_1, (16, ), (1, ))
    assert_size_stride(arg343_1, (16, ), (1, ))
    assert_size_stride(arg344_1, (16, ), (1, ))
    assert_size_stride(arg345_1, (16, ), (1, ))
    assert_size_stride(arg346_1, (64, ), (1, ))
    assert_size_stride(arg347_1, (64, ), (1, ))
    assert_size_stride(arg348_1, (64, ), (1, ))
    assert_size_stride(arg349_1, (64, ), (1, ))
    assert_size_stride(arg350_1, (24, ), (1, ))
    assert_size_stride(arg351_1, (24, ), (1, ))
    assert_size_stride(arg352_1, (48, ), (1, ))
    assert_size_stride(arg353_1, (48, ), (1, ))
    assert_size_stride(arg354_1, (48, ), (1, ))
    assert_size_stride(arg355_1, (48, ), (1, ))
    assert_size_stride(arg356_1, (24, ), (1, ))
    assert_size_stride(arg357_1, (24, ), (1, ))
    assert_size_stride(arg358_1, (48, ), (1, ))
    assert_size_stride(arg359_1, (48, ), (1, ))
    assert_size_stride(arg360_1, (48, ), (1, ))
    assert_size_stride(arg361_1, (48, ), (1, ))
    assert_size_stride(arg362_1, (24, ), (1, ))
    assert_size_stride(arg363_1, (24, ), (1, ))
    assert_size_stride(arg364_1, (48, ), (1, ))
    assert_size_stride(arg365_1, (48, ), (1, ))
    assert_size_stride(arg366_1, (48, ), (1, ))
    assert_size_stride(arg367_1, (48, ), (1, ))
    assert_size_stride(arg368_1, (24, ), (1, ))
    assert_size_stride(arg369_1, (24, ), (1, ))
    assert_size_stride(arg370_1, (120, ), (1, ))
    assert_size_stride(arg371_1, (120, ), (1, ))
    assert_size_stride(arg372_1, (120, ), (1, ))
    assert_size_stride(arg373_1, (120, ), (1, ))
    assert_size_stride(arg374_1, (40, ), (1, ))
    assert_size_stride(arg375_1, (40, ), (1, ))
    assert_size_stride(arg376_1, (120, ), (1, ))
    assert_size_stride(arg377_1, (120, ), (1, ))
    assert_size_stride(arg378_1, (120, ), (1, ))
    assert_size_stride(arg379_1, (120, ), (1, ))
    assert_size_stride(arg380_1, (40, ), (1, ))
    assert_size_stride(arg381_1, (40, ), (1, ))
    assert_size_stride(arg382_1, (120, ), (1, ))
    assert_size_stride(arg383_1, (120, ), (1, ))
    assert_size_stride(arg384_1, (120, ), (1, ))
    assert_size_stride(arg385_1, (120, ), (1, ))
    assert_size_stride(arg386_1, (40, ), (1, ))
    assert_size_stride(arg387_1, (40, ), (1, ))
    assert_size_stride(arg388_1, (120, ), (1, ))
    assert_size_stride(arg389_1, (120, ), (1, ))
    assert_size_stride(arg390_1, (120, ), (1, ))
    assert_size_stride(arg391_1, (120, ), (1, ))
    assert_size_stride(arg392_1, (40, ), (1, ))
    assert_size_stride(arg393_1, (40, ), (1, ))
    assert_size_stride(arg394_1, (120, ), (1, ))
    assert_size_stride(arg395_1, (120, ), (1, ))
    assert_size_stride(arg396_1, (120, ), (1, ))
    assert_size_stride(arg397_1, (120, ), (1, ))
    assert_size_stride(arg398_1, (40, ), (1, ))
    assert_size_stride(arg399_1, (40, ), (1, ))
    assert_size_stride(arg400_1, (200, ), (1, ))
    assert_size_stride(arg401_1, (200, ), (1, ))
    assert_size_stride(arg402_1, (200, ), (1, ))
    assert_size_stride(arg403_1, (200, ), (1, ))
    assert_size_stride(arg404_1, (72, ), (1, ))
    assert_size_stride(arg405_1, (72, ), (1, ))
    assert_size_stride(arg406_1, (216, ), (1, ))
    assert_size_stride(arg407_1, (216, ), (1, ))
    assert_size_stride(arg408_1, (216, ), (1, ))
    assert_size_stride(arg409_1, (216, ), (1, ))
    assert_size_stride(arg410_1, (72, ), (1, ))
    assert_size_stride(arg411_1, (72, ), (1, ))
    assert_size_stride(arg412_1, (216, ), (1, ))
    assert_size_stride(arg413_1, (216, ), (1, ))
    assert_size_stride(arg414_1, (216, ), (1, ))
    assert_size_stride(arg415_1, (216, ), (1, ))
    assert_size_stride(arg416_1, (72, ), (1, ))
    assert_size_stride(arg417_1, (72, ), (1, ))
    assert_size_stride(arg418_1, (216, ), (1, ))
    assert_size_stride(arg419_1, (216, ), (1, ))
    assert_size_stride(arg420_1, (216, ), (1, ))
    assert_size_stride(arg421_1, (216, ), (1, ))
    assert_size_stride(arg422_1, (72, ), (1, ))
    assert_size_stride(arg423_1, (72, ), (1, ))
    assert_size_stride(arg424_1, (216, ), (1, ))
    assert_size_stride(arg425_1, (216, ), (1, ))
    assert_size_stride(arg426_1, (216, ), (1, ))
    assert_size_stride(arg427_1, (216, ), (1, ))
    assert_size_stride(arg428_1, (72, ), (1, ))
    assert_size_stride(arg429_1, (72, ), (1, ))
    assert_size_stride(arg430_1, (360, ), (1, ))
    assert_size_stride(arg431_1, (360, ), (1, ))
    assert_size_stride(arg432_1, (360, ), (1, ))
    assert_size_stride(arg433_1, (360, ), (1, ))
    assert_size_stride(arg434_1, (120, ), (1, ))
    assert_size_stride(arg435_1, (120, ), (1, ))
    assert_size_stride(arg436_1, (360, ), (1, ))
    assert_size_stride(arg437_1, (360, ), (1, ))
    assert_size_stride(arg438_1, (360, ), (1, ))
    assert_size_stride(arg439_1, (360, ), (1, ))
    assert_size_stride(arg440_1, (120, ), (1, ))
    assert_size_stride(arg441_1, (120, ), (1, ))
    assert_size_stride(arg442_1, (360, ), (1, ))
    assert_size_stride(arg443_1, (360, ), (1, ))
    assert_size_stride(arg444_1, (360, ), (1, ))
    assert_size_stride(arg445_1, (360, ), (1, ))
    assert_size_stride(arg446_1, (120, ), (1, ))
    assert_size_stride(arg447_1, (120, ), (1, ))
    assert_size_stride(arg448_1, (360, ), (1, ))
    assert_size_stride(arg449_1, (360, ), (1, ))
    assert_size_stride(arg450_1, (360, ), (1, ))
    assert_size_stride(arg451_1, (360, ), (1, ))
    assert_size_stride(arg452_1, (120, ), (1, ))
    assert_size_stride(arg453_1, (120, ), (1, ))
    assert_size_stride(arg454_1, (360, ), (1, ))
    assert_size_stride(arg455_1, (360, ), (1, ))
    assert_size_stride(arg456_1, (360, ), (1, ))
    assert_size_stride(arg457_1, (360, ), (1, ))
    assert_size_stride(arg458_1, (120, ), (1, ))
    assert_size_stride(arg459_1, (120, ), (1, ))
    assert_size_stride(arg460_1, (360, ), (1, ))
    assert_size_stride(arg461_1, (360, ), (1, ))
    assert_size_stride(arg462_1, (360, ), (1, ))
    assert_size_stride(arg463_1, (360, ), (1, ))
    assert_size_stride(arg464_1, (120, ), (1, ))
    assert_size_stride(arg465_1, (120, ), (1, ))
    assert_size_stride(arg466_1, (720, ), (1, ))
    assert_size_stride(arg467_1, (720, ), (1, ))
    assert_size_stride(arg468_1, (720, ), (1, ))
    assert_size_stride(arg469_1, (720, ), (1, ))
    assert_size_stride(arg470_1, (184, ), (1, ))
    assert_size_stride(arg471_1, (184, ), (1, ))
    assert_size_stride(arg472_1, (736, ), (1, ))
    assert_size_stride(arg473_1, (736, ), (1, ))
    assert_size_stride(arg474_1, (736, ), (1, ))
    assert_size_stride(arg475_1, (736, ), (1, ))
    assert_size_stride(arg476_1, (184, ), (1, ))
    assert_size_stride(arg477_1, (184, ), (1, ))
    assert_size_stride(arg478_1, (736, ), (1, ))
    assert_size_stride(arg479_1, (736, ), (1, ))
    assert_size_stride(arg480_1, (736, ), (1, ))
    assert_size_stride(arg481_1, (736, ), (1, ))
    assert_size_stride(arg482_1, (184, ), (1, ))
    assert_size_stride(arg483_1, (184, ), (1, ))
    assert_size_stride(arg484_1, (736, ), (1, ))
    assert_size_stride(arg485_1, (736, ), (1, ))
    assert_size_stride(arg486_1, (736, ), (1, ))
    assert_size_stride(arg487_1, (736, ), (1, ))
    assert_size_stride(arg488_1, (184, ), (1, ))
    assert_size_stride(arg489_1, (184, ), (1, ))
    assert_size_stride(arg490_1, (736, ), (1, ))
    assert_size_stride(arg491_1, (736, ), (1, ))
    assert_size_stride(arg492_1, (736, ), (1, ))
    assert_size_stride(arg493_1, (736, ), (1, ))
    assert_size_stride(arg494_1, (184, ), (1, ))
    assert_size_stride(arg495_1, (184, ), (1, ))
    assert_size_stride(arg496_1, (736, ), (1, ))
    assert_size_stride(arg497_1, (736, ), (1, ))
    assert_size_stride(arg498_1, (736, ), (1, ))
    assert_size_stride(arg499_1, (736, ), (1, ))
    assert_size_stride(arg500_1, (184, ), (1, ))
    assert_size_stride(arg501_1, (184, ), (1, ))
    assert_size_stride(arg502_1, (1104, ), (1, ))
    assert_size_stride(arg503_1, (1104, ), (1, ))
    assert_size_stride(arg504_1, (1104, ), (1, ))
    assert_size_stride(arg505_1, (1104, ), (1, ))
    assert_size_stride(arg506_1, (224, ), (1, ))
    assert_size_stride(arg507_1, (224, ), (1, ))
    assert_size_stride(arg508_1, (1344, ), (1, ))
    assert_size_stride(arg509_1, (1344, ), (1, ))
    assert_size_stride(arg510_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg510_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg510_1
        buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg176_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg176_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 16, 128, 128), (262144, 1, 2048, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf3, arg336_1, arg337_1, arg0_1, arg1_1, buf4, 128, 16384, grid=grid(128, 16384), stream=stream0)
        del arg0_1
        del arg1_1
        del arg336_1
        del arg337_1
        # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.hardswish]
        buf5 = extern_kernels.convolution(buf4, arg177_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf5, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del arg177_1
        buf6 = buf5; del buf5  # reuse
        buf7 = reinterpret_tensor(buf3, (8, 16, 128, 128), (262144, 1, 2048, 16), 0); del buf3  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf6, arg338_1, arg339_1, arg2_1, arg3_1, buf7, 128, 16384, grid=grid(128, 16384), stream=stream0)
        del arg2_1
        del arg338_1
        del arg339_1
        del arg3_1
        del buf6
        # Source Nodes: [x_11, x_9], Original ATen: [aten.convolution, aten.hardswish]
        buf8 = extern_kernels.convolution(buf7, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del arg178_1
        del buf7
        buf9 = buf4; del buf4  # reuse
        # Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf9, buf8, arg340_1, arg341_1, arg4_1, arg5_1, 131072, 16, grid=grid(131072, 16), stream=stream0)
        del arg340_1
        del arg341_1
        del arg4_1
        del arg5_1
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg179_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf10, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del arg179_1
        buf11 = buf10; del buf10  # reuse
        buf12 = reinterpret_tensor(buf8, (8, 16, 128, 128), (262144, 1, 2048, 16), 0); del buf8  # reuse
        # Source Nodes: [x_18, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf11, arg342_1, arg343_1, arg6_1, arg7_1, buf12, 128, 16384, grid=grid(128, 16384), stream=stream0)
        del arg342_1
        del arg343_1
        del arg6_1
        del arg7_1
        del buf11
        # Source Nodes: [x_21, x_23], Original ATen: [aten.convolution, aten.hardswish]
        buf13 = extern_kernels.convolution(buf12, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del arg180_1
        del buf12
        buf14 = buf9; del buf9  # reuse
        # Source Nodes: [shortcut_2, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf14, buf13, arg344_1, arg345_1, arg8_1, arg9_1, 131072, 16, grid=grid(131072, 16), stream=stream0)
        del arg344_1
        del arg345_1
        del arg8_1
        del arg9_1
        del buf13
        # Source Nodes: [shortcut_2, x_24, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg181_1
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4.run(buf16, arg346_1, arg347_1, arg10_1, arg11_1, buf17, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg10_1
        del arg11_1
        del arg346_1
        del arg347_1
        del buf16
        # Source Nodes: [x_33, x_34], Original ATen: [aten.convolution, aten.hardswish]
        buf18 = extern_kernels.convolution(buf17, arg182_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf18, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg182_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        buf20 = reinterpret_tensor(buf14, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf14  # reuse
        # Source Nodes: [x_35, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf19, arg348_1, arg349_1, arg12_1, arg13_1, buf20, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg12_1
        del arg13_1
        del arg348_1
        del arg349_1
        del buf19
        # Source Nodes: [x_38, x_40], Original ATen: [aten.convolution, aten.hardswish]
        buf21 = extern_kernels.convolution(buf20, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 24, 64, 64), (98304, 4096, 64, 1))
        del arg183_1
        del buf20
        buf22 = empty_strided((8, 24, 64, 64), (98304, 1, 1536, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf21, arg350_1, arg351_1, arg14_1, arg15_1, buf22, 192, 4096, grid=grid(192, 4096), stream=stream0)
        del arg14_1
        del arg15_1
        del arg350_1
        del arg351_1
        del buf21
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 48, 64, 64), (196608, 4096, 64, 1))
        del arg184_1
        buf24 = buf23; del buf23  # reuse
        buf25 = reinterpret_tensor(buf0, (8, 48, 64, 64), (196608, 1, 3072, 48), 0); del buf0  # reuse
        # Source Nodes: [x_46, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf24, arg352_1, arg353_1, arg16_1, arg17_1, buf25, 384, 4096, grid=grid(384, 4096), stream=stream0)
        del arg16_1
        del arg17_1
        del arg352_1
        del arg353_1
        del buf24
        # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.hardswish]
        buf26 = extern_kernels.convolution(buf25, arg185_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf26, (8, 48, 64, 64), (196608, 4096, 64, 1))
        del arg185_1
        buf27 = buf26; del buf26  # reuse
        buf28 = buf25; del buf25  # reuse
        # Source Nodes: [x_51, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf27, arg354_1, arg355_1, arg18_1, arg19_1, buf28, 384, 4096, grid=grid(384, 4096), stream=stream0)
        del arg18_1
        del arg19_1
        del arg354_1
        del arg355_1
        del buf27
        # Source Nodes: [x_54, x_56], Original ATen: [aten.convolution, aten.hardswish]
        buf29 = extern_kernels.convolution(buf28, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 24, 64, 64), (98304, 4096, 64, 1))
        del arg186_1
        buf30 = buf22; del buf22  # reuse
        # Source Nodes: [shortcut_4, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf30, buf29, arg356_1, arg357_1, arg20_1, arg21_1, 32768, 24, grid=grid(32768, 24), stream=stream0)
        del arg20_1
        del arg21_1
        del arg356_1
        del arg357_1
        del buf29
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 48, 64, 64), (196608, 4096, 64, 1))
        del arg187_1
        buf32 = buf31; del buf31  # reuse
        buf33 = buf28; del buf28  # reuse
        # Source Nodes: [x_63, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf32, arg358_1, arg359_1, arg22_1, arg23_1, buf33, 384, 4096, grid=grid(384, 4096), stream=stream0)
        del arg22_1
        del arg23_1
        del arg358_1
        del arg359_1
        del buf32
        # Source Nodes: [x_66, x_67], Original ATen: [aten.convolution, aten.hardswish]
        buf34 = extern_kernels.convolution(buf33, arg188_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf34, (8, 48, 64, 64), (196608, 4096, 64, 1))
        del arg188_1
        buf35 = buf34; del buf34  # reuse
        buf36 = buf33; del buf33  # reuse
        # Source Nodes: [x_68, x_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf35, arg360_1, arg361_1, arg24_1, arg25_1, buf36, 384, 4096, grid=grid(384, 4096), stream=stream0)
        del arg24_1
        del arg25_1
        del arg360_1
        del arg361_1
        del buf35
        # Source Nodes: [x_71, x_73], Original ATen: [aten.convolution, aten.hardswish]
        buf37 = extern_kernels.convolution(buf36, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 24, 64, 64), (98304, 4096, 64, 1))
        del arg189_1
        buf38 = buf30; del buf30  # reuse
        # Source Nodes: [shortcut_5, x_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf38, buf37, arg362_1, arg363_1, arg26_1, arg27_1, 32768, 24, grid=grid(32768, 24), stream=stream0)
        del arg26_1
        del arg27_1
        del arg362_1
        del arg363_1
        del buf37
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 48, 64, 64), (196608, 4096, 64, 1))
        del arg190_1
        buf40 = buf39; del buf39  # reuse
        buf41 = buf36; del buf36  # reuse
        # Source Nodes: [x_80, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf40, arg364_1, arg365_1, arg28_1, arg29_1, buf41, 384, 4096, grid=grid(384, 4096), stream=stream0)
        del arg28_1
        del arg29_1
        del arg364_1
        del arg365_1
        del buf40
        # Source Nodes: [x_83, x_84], Original ATen: [aten.convolution, aten.hardswish]
        buf42 = extern_kernels.convolution(buf41, arg191_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf42, (8, 48, 64, 64), (196608, 4096, 64, 1))
        del arg191_1
        buf43 = buf42; del buf42  # reuse
        buf44 = buf41; del buf41  # reuse
        # Source Nodes: [x_85, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf43, arg366_1, arg367_1, arg30_1, arg31_1, buf44, 384, 4096, grid=grid(384, 4096), stream=stream0)
        del arg30_1
        del arg31_1
        del arg366_1
        del arg367_1
        del buf43
        # Source Nodes: [x_88, x_90], Original ATen: [aten.convolution, aten.hardswish]
        buf45 = extern_kernels.convolution(buf44, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 24, 64, 64), (98304, 4096, 64, 1))
        del arg192_1
        del buf44
        buf46 = buf38; del buf38  # reuse
        # Source Nodes: [shortcut_6, x_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf46, buf45, arg368_1, arg369_1, arg32_1, arg33_1, 32768, 24, grid=grid(32768, 24), stream=stream0)
        del arg32_1
        del arg33_1
        del arg368_1
        del arg369_1
        del buf45
        # Source Nodes: [shortcut_6, x_91, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 120, 64, 64), (491520, 4096, 64, 1))
        del arg193_1
        del buf46
        buf48 = buf47; del buf47  # reuse
        buf49 = empty_strided((8, 120, 64, 64), (491520, 1, 7680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf48, arg370_1, arg371_1, arg34_1, arg35_1, buf49, 960, 4096, grid=grid(960, 4096), stream=stream0)
        del arg34_1
        del arg35_1
        del arg370_1
        del arg371_1
        del buf48
        # Source Nodes: [x_100, x_101], Original ATen: [aten.convolution, aten.hardswish]
        buf50 = extern_kernels.convolution(buf49, arg194_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf50, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg194_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        buf52 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf53 = reinterpret_tensor(buf52, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf52  # reuse
        # Source Nodes: [x_102, x_105, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10.run(buf51, buf53, arg372_1, arg373_1, arg36_1, arg37_1, 960, 1024, grid=grid(960), stream=stream0)
        del arg36_1
        del arg372_1
        del arg373_1
        del arg37_1
        # Source Nodes: [x_105, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf54 = extern_kernels.convolution(buf53, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg195_1
        del buf53
        buf55 = reinterpret_tensor(buf54, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf54  # reuse
        # Source Nodes: [x_105, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_11.run(buf55, arg196_1, 64, grid=grid(64), stream=stream0)
        del arg196_1
        # Source Nodes: [x_105, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf56 = extern_kernels.convolution(buf55, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg197_1
        del buf55
        buf57 = empty_strided((8, 120, 32, 32), (122880, 1, 3840, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_105, x_106, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_12.run(buf51, buf56, arg198_1, buf57, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg198_1
        del buf51
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_105, x_106, x_107, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf58 = extern_kernels.convolution(buf57, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 40, 32, 32), (40960, 1024, 32, 1))
        del arg199_1
        buf59 = empty_strided((8, 40, 32, 32), (40960, 1, 1280, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf58, arg374_1, arg375_1, arg38_1, arg39_1, buf59, 320, 1024, grid=grid(320, 1024), stream=stream0)
        del arg374_1
        del arg375_1
        del arg38_1
        del arg39_1
        del buf58
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg200_1
        buf61 = buf60; del buf60  # reuse
        buf62 = buf57; del buf57  # reuse
        # Source Nodes: [x_113, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14.run(buf61, arg376_1, arg377_1, arg40_1, arg41_1, buf62, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg376_1
        del arg377_1
        del arg40_1
        del arg41_1
        del buf61
        # Source Nodes: [x_116, x_117], Original ATen: [aten.convolution, aten.hardswish]
        buf63 = extern_kernels.convolution(buf62, arg201_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf63, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg201_1
        buf64 = buf63; del buf63  # reuse
        buf65 = reinterpret_tensor(buf56, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf56  # reuse
        buf66 = reinterpret_tensor(buf65, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf65  # reuse
        # Source Nodes: [x_118, x_121, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10.run(buf64, buf66, arg378_1, arg379_1, arg42_1, arg43_1, 960, 1024, grid=grid(960), stream=stream0)
        del arg378_1
        del arg379_1
        del arg42_1
        del arg43_1
        # Source Nodes: [x_121, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf67 = extern_kernels.convolution(buf66, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg202_1
        del buf66
        buf68 = reinterpret_tensor(buf67, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf67  # reuse
        # Source Nodes: [x_121, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_15.run(buf68, arg203_1, 128, grid=grid(128), stream=stream0)
        del arg203_1
        # Source Nodes: [x_121, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf69 = extern_kernels.convolution(buf68, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg204_1
        del buf68
        buf70 = buf62; del buf62  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_121, x_122, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_12.run(buf64, buf69, arg205_1, buf70, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg205_1
        del buf64
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_121, x_122, x_123, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf71 = extern_kernels.convolution(buf70, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 40, 32, 32), (40960, 1024, 32, 1))
        del arg206_1
        buf72 = buf59; del buf59  # reuse
        # Source Nodes: [shortcut_8, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf72, buf71, arg380_1, arg381_1, arg44_1, arg45_1, 8192, 40, grid=grid(8192, 40), stream=stream0)
        del arg380_1
        del arg381_1
        del arg44_1
        del arg45_1
        del buf71
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg207_1
        buf74 = buf73; del buf73  # reuse
        buf75 = buf70; del buf70  # reuse
        # Source Nodes: [x_130, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14.run(buf74, arg382_1, arg383_1, arg46_1, arg47_1, buf75, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg382_1
        del arg383_1
        del arg46_1
        del arg47_1
        del buf74
        # Source Nodes: [x_133, x_134], Original ATen: [aten.convolution, aten.hardswish]
        buf76 = extern_kernels.convolution(buf75, arg208_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf76, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg208_1
        buf77 = buf76; del buf76  # reuse
        buf78 = reinterpret_tensor(buf69, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf69  # reuse
        buf79 = reinterpret_tensor(buf78, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf78  # reuse
        # Source Nodes: [x_135, x_138, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10.run(buf77, buf79, arg384_1, arg385_1, arg48_1, arg49_1, 960, 1024, grid=grid(960), stream=stream0)
        del arg384_1
        del arg385_1
        del arg48_1
        del arg49_1
        # Source Nodes: [x_138, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf80 = extern_kernels.convolution(buf79, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg209_1
        del buf79
        buf81 = reinterpret_tensor(buf80, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf80  # reuse
        # Source Nodes: [x_138, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_15.run(buf81, arg210_1, 128, grid=grid(128), stream=stream0)
        del arg210_1
        # Source Nodes: [x_138, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf82 = extern_kernels.convolution(buf81, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg211_1
        del buf81
        buf83 = buf75; del buf75  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_138, x_139, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_12.run(buf77, buf82, arg212_1, buf83, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg212_1
        del buf77
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_138, x_139, x_140, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf84 = extern_kernels.convolution(buf83, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 40, 32, 32), (40960, 1024, 32, 1))
        del arg213_1
        buf85 = buf72; del buf72  # reuse
        # Source Nodes: [shortcut_9, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf85, buf84, arg386_1, arg387_1, arg50_1, arg51_1, 8192, 40, grid=grid(8192, 40), stream=stream0)
        del arg386_1
        del arg387_1
        del arg50_1
        del arg51_1
        del buf84
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg214_1
        buf87 = buf86; del buf86  # reuse
        buf88 = buf83; del buf83  # reuse
        # Source Nodes: [x_147, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14.run(buf87, arg388_1, arg389_1, arg52_1, arg53_1, buf88, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg388_1
        del arg389_1
        del arg52_1
        del arg53_1
        del buf87
        # Source Nodes: [x_150, x_151], Original ATen: [aten.convolution, aten.hardswish]
        buf89 = extern_kernels.convolution(buf88, arg215_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf89, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg215_1
        buf90 = buf89; del buf89  # reuse
        buf91 = reinterpret_tensor(buf82, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf82  # reuse
        buf92 = reinterpret_tensor(buf91, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf91  # reuse
        # Source Nodes: [x_152, x_155, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10.run(buf90, buf92, arg390_1, arg391_1, arg54_1, arg55_1, 960, 1024, grid=grid(960), stream=stream0)
        del arg390_1
        del arg391_1
        del arg54_1
        del arg55_1
        # Source Nodes: [x_155, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf93 = extern_kernels.convolution(buf92, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg216_1
        del buf92
        buf94 = reinterpret_tensor(buf93, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf93  # reuse
        # Source Nodes: [x_155, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_15.run(buf94, arg217_1, 128, grid=grid(128), stream=stream0)
        del arg217_1
        # Source Nodes: [x_155, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf95 = extern_kernels.convolution(buf94, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg218_1
        del buf94
        buf96 = buf88; del buf88  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_155, x_156, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_12.run(buf90, buf95, arg219_1, buf96, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg219_1
        del buf90
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_155, x_156, x_157, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf97 = extern_kernels.convolution(buf96, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 40, 32, 32), (40960, 1024, 32, 1))
        del arg220_1
        buf98 = buf85; del buf85  # reuse
        # Source Nodes: [shortcut_10, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf98, buf97, arg392_1, arg393_1, arg56_1, arg57_1, 8192, 40, grid=grid(8192, 40), stream=stream0)
        del arg392_1
        del arg393_1
        del arg56_1
        del arg57_1
        del buf97
        # Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg221_1
        buf100 = buf99; del buf99  # reuse
        buf101 = buf96; del buf96  # reuse
        # Source Nodes: [x_164, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14.run(buf100, arg394_1, arg395_1, arg58_1, arg59_1, buf101, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg394_1
        del arg395_1
        del arg58_1
        del arg59_1
        del buf100
        # Source Nodes: [x_167, x_168], Original ATen: [aten.convolution, aten.hardswish]
        buf102 = extern_kernels.convolution(buf101, arg222_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf102, (8, 120, 32, 32), (122880, 1024, 32, 1))
        del arg222_1
        buf103 = buf102; del buf102  # reuse
        buf104 = reinterpret_tensor(buf95, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf95  # reuse
        buf105 = reinterpret_tensor(buf104, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf104  # reuse
        # Source Nodes: [x_169, x_172, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10.run(buf103, buf105, arg396_1, arg397_1, arg60_1, arg61_1, 960, 1024, grid=grid(960), stream=stream0)
        del arg396_1
        del arg397_1
        del arg60_1
        del arg61_1
        # Source Nodes: [x_172, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf106 = extern_kernels.convolution(buf105, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg223_1
        del buf105
        buf107 = reinterpret_tensor(buf106, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf106  # reuse
        # Source Nodes: [x_172, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_15.run(buf107, arg224_1, 128, grid=grid(128), stream=stream0)
        del arg224_1
        # Source Nodes: [x_172, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf108 = extern_kernels.convolution(buf107, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg225_1
        del buf107
        buf109 = buf101; del buf101  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____4___se_gate, x_172, x_173, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_12.run(buf103, buf108, arg226_1, buf109, 960, 1024, grid=grid(960, 1024), stream=stream0)
        del arg226_1
        del buf103
        del buf108
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____4___se_gate, x_172, x_173, x_174, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf110 = extern_kernels.convolution(buf109, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 40, 32, 32), (40960, 1024, 32, 1))
        del arg227_1
        del buf109
        buf111 = buf98; del buf98  # reuse
        # Source Nodes: [shortcut_11, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf111, buf110, arg398_1, arg399_1, arg62_1, arg63_1, 8192, 40, grid=grid(8192, 40), stream=stream0)
        del arg398_1
        del arg399_1
        del arg62_1
        del arg63_1
        del buf110
        # Source Nodes: [shortcut_11, x_175, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 200, 32, 32), (204800, 1024, 32, 1))
        del arg228_1
        del buf111
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided((8, 200, 32, 32), (204800, 1, 6400, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17.run(buf113, arg400_1, arg401_1, arg64_1, arg65_1, buf114, 1600, 1024, grid=grid(1600, 1024), stream=stream0)
        del arg400_1
        del arg401_1
        del arg64_1
        del arg65_1
        del buf113
        # Source Nodes: [x_184, x_185], Original ATen: [aten.convolution, aten.hardswish]
        buf115 = extern_kernels.convolution(buf114, arg229_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf115, (8, 200, 16, 16), (51200, 256, 16, 1))
        del arg229_1
        del buf114
        buf116 = buf115; del buf115  # reuse
        buf117 = empty_strided((8, 200, 16, 16), (51200, 1, 3200, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_18.run(buf116, arg402_1, arg403_1, arg66_1, arg67_1, buf117, 1600, 256, grid=grid(1600, 256), stream=stream0)
        del arg402_1
        del arg403_1
        del arg66_1
        del arg67_1
        del buf116
        # Source Nodes: [x_189, x_191], Original ATen: [aten.convolution, aten.hardswish]
        buf118 = extern_kernels.convolution(buf117, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 72, 16, 16), (18432, 256, 16, 1))
        del arg230_1
        del buf117
        buf119 = empty_strided((8, 72, 16, 16), (18432, 1, 1152, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf118, arg404_1, arg405_1, arg68_1, arg69_1, buf119, 576, 256, grid=grid(576, 256), stream=stream0)
        del arg404_1
        del arg405_1
        del arg68_1
        del arg69_1
        del buf118
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg231_1
        buf121 = buf120; del buf120  # reuse
        buf122 = empty_strided((8, 216, 16, 16), (55296, 1, 3456, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf121, arg406_1, arg407_1, arg70_1, arg71_1, buf122, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg406_1
        del arg407_1
        del arg70_1
        del arg71_1
        del buf121
        # Source Nodes: [x_200, x_201], Original ATen: [aten.convolution, aten.hardswish]
        buf123 = extern_kernels.convolution(buf122, arg232_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf123, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg232_1
        buf124 = buf123; del buf123  # reuse
        buf125 = buf122; del buf122  # reuse
        # Source Nodes: [x_202, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf124, arg408_1, arg409_1, arg72_1, arg73_1, buf125, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg408_1
        del arg409_1
        del arg72_1
        del arg73_1
        del buf124
        # Source Nodes: [x_205, x_207], Original ATen: [aten.convolution, aten.hardswish]
        buf126 = extern_kernels.convolution(buf125, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 72, 16, 16), (18432, 256, 16, 1))
        del arg233_1
        buf127 = buf119; del buf119  # reuse
        # Source Nodes: [shortcut_13, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf127, buf126, arg410_1, arg411_1, arg74_1, arg75_1, 2048, 72, grid=grid(2048, 72), stream=stream0)
        del arg410_1
        del arg411_1
        del arg74_1
        del arg75_1
        del buf126
        # Source Nodes: [x_213], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg234_1
        buf129 = buf128; del buf128  # reuse
        buf130 = buf125; del buf125  # reuse
        # Source Nodes: [x_214, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf129, arg412_1, arg413_1, arg76_1, arg77_1, buf130, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg412_1
        del arg413_1
        del arg76_1
        del arg77_1
        del buf129
        # Source Nodes: [x_217, x_218], Original ATen: [aten.convolution, aten.hardswish]
        buf131 = extern_kernels.convolution(buf130, arg235_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf131, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg235_1
        buf132 = buf131; del buf131  # reuse
        buf133 = buf130; del buf130  # reuse
        # Source Nodes: [x_219, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf132, arg414_1, arg415_1, arg78_1, arg79_1, buf133, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg414_1
        del arg415_1
        del arg78_1
        del arg79_1
        del buf132
        # Source Nodes: [x_222, x_224], Original ATen: [aten.convolution, aten.hardswish]
        buf134 = extern_kernels.convolution(buf133, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 72, 16, 16), (18432, 256, 16, 1))
        del arg236_1
        buf135 = buf127; del buf127  # reuse
        # Source Nodes: [shortcut_14, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf135, buf134, arg416_1, arg417_1, arg80_1, arg81_1, 2048, 72, grid=grid(2048, 72), stream=stream0)
        del arg416_1
        del arg417_1
        del arg80_1
        del arg81_1
        del buf134
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg237_1
        buf137 = buf136; del buf136  # reuse
        buf138 = buf133; del buf133  # reuse
        # Source Nodes: [x_231, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf137, arg418_1, arg419_1, arg82_1, arg83_1, buf138, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg418_1
        del arg419_1
        del arg82_1
        del arg83_1
        del buf137
        # Source Nodes: [x_234, x_235], Original ATen: [aten.convolution, aten.hardswish]
        buf139 = extern_kernels.convolution(buf138, arg238_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf139, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg238_1
        buf140 = buf139; del buf139  # reuse
        buf141 = buf138; del buf138  # reuse
        # Source Nodes: [x_236, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf140, arg420_1, arg421_1, arg84_1, arg85_1, buf141, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg420_1
        del arg421_1
        del arg84_1
        del arg85_1
        del buf140
        # Source Nodes: [x_239, x_241], Original ATen: [aten.convolution, aten.hardswish]
        buf142 = extern_kernels.convolution(buf141, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 72, 16, 16), (18432, 256, 16, 1))
        del arg239_1
        buf143 = buf135; del buf135  # reuse
        # Source Nodes: [shortcut_15, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf143, buf142, arg422_1, arg423_1, arg86_1, arg87_1, 2048, 72, grid=grid(2048, 72), stream=stream0)
        del arg422_1
        del arg423_1
        del arg86_1
        del arg87_1
        del buf142
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg240_1
        buf145 = buf144; del buf144  # reuse
        buf146 = buf141; del buf141  # reuse
        # Source Nodes: [x_248, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf145, arg424_1, arg425_1, arg88_1, arg89_1, buf146, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg424_1
        del arg425_1
        del arg88_1
        del arg89_1
        del buf145
        # Source Nodes: [x_251, x_252], Original ATen: [aten.convolution, aten.hardswish]
        buf147 = extern_kernels.convolution(buf146, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf147, (8, 216, 16, 16), (55296, 256, 16, 1))
        del arg241_1
        buf148 = buf147; del buf147  # reuse
        buf149 = buf146; del buf146  # reuse
        # Source Nodes: [x_253, x_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf148, arg426_1, arg427_1, arg90_1, arg91_1, buf149, 1728, 256, grid=grid(1728, 256), stream=stream0)
        del arg426_1
        del arg427_1
        del arg90_1
        del arg91_1
        del buf148
        # Source Nodes: [x_256, x_258], Original ATen: [aten.convolution, aten.hardswish]
        buf150 = extern_kernels.convolution(buf149, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 72, 16, 16), (18432, 256, 16, 1))
        del arg242_1
        del buf149
        buf151 = buf143; del buf143  # reuse
        # Source Nodes: [shortcut_16, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf151, buf150, arg428_1, arg429_1, arg92_1, arg93_1, 2048, 72, grid=grid(2048, 72), stream=stream0)
        del arg428_1
        del arg429_1
        del arg92_1
        del arg93_1
        del buf150
        # Source Nodes: [shortcut_16, x_259, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg243_1
        del buf151
        buf153 = buf152; del buf152  # reuse
        buf154 = empty_strided((8, 360, 16, 16), (92160, 1, 5760, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_265, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf153, arg430_1, arg431_1, arg94_1, arg95_1, buf154, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg430_1
        del arg431_1
        del arg94_1
        del arg95_1
        del buf153
        # Source Nodes: [x_268, x_269], Original ATen: [aten.convolution, aten.hardswish]
        buf155 = extern_kernels.convolution(buf154, arg244_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf155, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg244_1
        buf156 = buf155; del buf155  # reuse
        buf157 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cuda', dtype=torch.float32)
        buf158 = reinterpret_tensor(buf157, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf157  # reuse
        # Source Nodes: [x_270, x_273, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23.run(buf156, buf158, arg432_1, arg433_1, arg96_1, arg97_1, 2880, 256, grid=grid(2880), stream=stream0)
        del arg432_1
        del arg433_1
        del arg96_1
        del arg97_1
        # Source Nodes: [x_273, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf159 = extern_kernels.convolution(buf158, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 24, 1, 1), (24, 1, 1, 1))
        del arg245_1
        del buf158
        buf160 = reinterpret_tensor(buf159, (8, 24, 1, 1), (24, 1, 24, 24), 0); del buf159  # reuse
        # Source Nodes: [x_273, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_24.run(buf160, arg246_1, 192, grid=grid(192), stream=stream0)
        del arg246_1
        # Source Nodes: [x_273, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf161 = extern_kernels.convolution(buf160, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg247_1
        del buf160
        buf162 = buf154; del buf154  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_273, x_274, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25.run(buf156, buf161, arg248_1, buf162, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg248_1
        del buf156
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_273, x_274, x_275, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf163 = extern_kernels.convolution(buf162, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 120, 16, 16), (30720, 256, 16, 1))
        del arg249_1
        buf164 = empty_strided((8, 120, 16, 16), (30720, 1, 1920, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_26.run(buf163, arg434_1, arg435_1, arg98_1, arg99_1, buf164, 960, 256, grid=grid(960, 256), stream=stream0)
        del arg434_1
        del arg435_1
        del arg98_1
        del arg99_1
        del buf163
        # Source Nodes: [x_280], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg250_1
        buf166 = buf165; del buf165  # reuse
        buf167 = buf162; del buf162  # reuse
        # Source Nodes: [x_281, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf166, arg436_1, arg437_1, arg100_1, arg101_1, buf167, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg100_1
        del arg101_1
        del arg436_1
        del arg437_1
        del buf166
        # Source Nodes: [x_284, x_285], Original ATen: [aten.convolution, aten.hardswish]
        buf168 = extern_kernels.convolution(buf167, arg251_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf168, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg251_1
        buf169 = buf168; del buf168  # reuse
        buf170 = reinterpret_tensor(buf161, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf161  # reuse
        buf171 = reinterpret_tensor(buf170, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf170  # reuse
        # Source Nodes: [x_286, x_289, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23.run(buf169, buf171, arg438_1, arg439_1, arg102_1, arg103_1, 2880, 256, grid=grid(2880), stream=stream0)
        del arg102_1
        del arg103_1
        del arg438_1
        del arg439_1
        # Source Nodes: [x_289, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf172 = extern_kernels.convolution(buf171, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg252_1
        del buf171
        buf173 = reinterpret_tensor(buf172, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf172  # reuse
        # Source Nodes: [x_289, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_27.run(buf173, arg253_1, 256, grid=grid(256), stream=stream0)
        del arg253_1
        # Source Nodes: [x_289, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf174 = extern_kernels.convolution(buf173, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg254_1
        del buf173
        buf175 = buf167; del buf167  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_289, x_290, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25.run(buf169, buf174, arg255_1, buf175, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg255_1
        del buf169
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_289, x_290, x_291, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf176 = extern_kernels.convolution(buf175, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 120, 16, 16), (30720, 256, 16, 1))
        del arg256_1
        buf177 = buf164; del buf164  # reuse
        # Source Nodes: [shortcut_18, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf177, buf176, arg440_1, arg441_1, arg104_1, arg105_1, 2048, 120, grid=grid(2048, 120), stream=stream0)
        del arg104_1
        del arg105_1
        del arg440_1
        del arg441_1
        del buf176
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg257_1
        buf179 = buf178; del buf178  # reuse
        buf180 = buf175; del buf175  # reuse
        # Source Nodes: [x_298, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf179, arg442_1, arg443_1, arg106_1, arg107_1, buf180, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg106_1
        del arg107_1
        del arg442_1
        del arg443_1
        del buf179
        # Source Nodes: [x_301, x_302], Original ATen: [aten.convolution, aten.hardswish]
        buf181 = extern_kernels.convolution(buf180, arg258_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf181, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg258_1
        buf182 = buf181; del buf181  # reuse
        buf183 = reinterpret_tensor(buf174, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf174  # reuse
        buf184 = reinterpret_tensor(buf183, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf183  # reuse
        # Source Nodes: [x_303, x_306, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23.run(buf182, buf184, arg444_1, arg445_1, arg108_1, arg109_1, 2880, 256, grid=grid(2880), stream=stream0)
        del arg108_1
        del arg109_1
        del arg444_1
        del arg445_1
        # Source Nodes: [x_306, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf185 = extern_kernels.convolution(buf184, arg259_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg259_1
        del buf184
        buf186 = reinterpret_tensor(buf185, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf185  # reuse
        # Source Nodes: [x_306, x_se_28, x_se_29, x_se_30], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_27.run(buf186, arg260_1, 256, grid=grid(256), stream=stream0)
        del arg260_1
        # Source Nodes: [x_306, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf187 = extern_kernels.convolution(buf186, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg261_1
        del buf186
        buf188 = buf180; del buf180  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_306, x_307, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25.run(buf182, buf187, arg262_1, buf188, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg262_1
        del buf182
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_306, x_307, x_308, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf189 = extern_kernels.convolution(buf188, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 120, 16, 16), (30720, 256, 16, 1))
        del arg263_1
        buf190 = buf177; del buf177  # reuse
        # Source Nodes: [shortcut_19, x_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf190, buf189, arg446_1, arg447_1, arg110_1, arg111_1, 2048, 120, grid=grid(2048, 120), stream=stream0)
        del arg110_1
        del arg111_1
        del arg446_1
        del arg447_1
        del buf189
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg264_1
        buf192 = buf191; del buf191  # reuse
        buf193 = buf188; del buf188  # reuse
        # Source Nodes: [x_315, x_318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf192, arg448_1, arg449_1, arg112_1, arg113_1, buf193, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg112_1
        del arg113_1
        del arg448_1
        del arg449_1
        del buf192
        # Source Nodes: [x_318, x_319], Original ATen: [aten.convolution, aten.hardswish]
        buf194 = extern_kernels.convolution(buf193, arg265_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf194, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg265_1
        buf195 = buf194; del buf194  # reuse
        buf196 = reinterpret_tensor(buf187, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf187  # reuse
        buf197 = reinterpret_tensor(buf196, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf196  # reuse
        # Source Nodes: [x_320, x_323, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23.run(buf195, buf197, arg450_1, arg451_1, arg114_1, arg115_1, 2880, 256, grid=grid(2880), stream=stream0)
        del arg114_1
        del arg115_1
        del arg450_1
        del arg451_1
        # Source Nodes: [x_323, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf198 = extern_kernels.convolution(buf197, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg266_1
        del buf197
        buf199 = reinterpret_tensor(buf198, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf198  # reuse
        # Source Nodes: [x_323, x_se_32, x_se_33, x_se_34], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_27.run(buf199, arg267_1, 256, grid=grid(256), stream=stream0)
        del arg267_1
        # Source Nodes: [x_323, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf200 = extern_kernels.convolution(buf199, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg268_1
        del buf199
        buf201 = buf193; del buf193  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_323, x_324, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25.run(buf195, buf200, arg269_1, buf201, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg269_1
        del buf195
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_323, x_324, x_325, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf202 = extern_kernels.convolution(buf201, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 120, 16, 16), (30720, 256, 16, 1))
        del arg270_1
        buf203 = buf190; del buf190  # reuse
        # Source Nodes: [shortcut_20, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf203, buf202, arg452_1, arg453_1, arg116_1, arg117_1, 2048, 120, grid=grid(2048, 120), stream=stream0)
        del arg116_1
        del arg117_1
        del arg452_1
        del arg453_1
        del buf202
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg271_1
        buf205 = buf204; del buf204  # reuse
        buf206 = buf201; del buf201  # reuse
        # Source Nodes: [x_332, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf205, arg454_1, arg455_1, arg118_1, arg119_1, buf206, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg118_1
        del arg119_1
        del arg454_1
        del arg455_1
        del buf205
        # Source Nodes: [x_335, x_336], Original ATen: [aten.convolution, aten.hardswish]
        buf207 = extern_kernels.convolution(buf206, arg272_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf207, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg272_1
        buf208 = buf207; del buf207  # reuse
        buf209 = reinterpret_tensor(buf200, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf200  # reuse
        buf210 = reinterpret_tensor(buf209, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf209  # reuse
        # Source Nodes: [x_337, x_340, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23.run(buf208, buf210, arg456_1, arg457_1, arg120_1, arg121_1, 2880, 256, grid=grid(2880), stream=stream0)
        del arg120_1
        del arg121_1
        del arg456_1
        del arg457_1
        # Source Nodes: [x_340, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf211 = extern_kernels.convolution(buf210, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg273_1
        del buf210
        buf212 = reinterpret_tensor(buf211, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf211  # reuse
        # Source Nodes: [x_340, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_27.run(buf212, arg274_1, 256, grid=grid(256), stream=stream0)
        del arg274_1
        # Source Nodes: [x_340, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf213 = extern_kernels.convolution(buf212, arg275_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg275_1
        del buf212
        buf214 = buf206; del buf206  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____4___se_gate, x_340, x_341, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25.run(buf208, buf213, arg276_1, buf214, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg276_1
        del buf208
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____4___se_gate, x_340, x_341, x_342, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf215 = extern_kernels.convolution(buf214, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 120, 16, 16), (30720, 256, 16, 1))
        del arg277_1
        buf216 = buf203; del buf203  # reuse
        # Source Nodes: [shortcut_21, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf216, buf215, arg458_1, arg459_1, arg122_1, arg123_1, 2048, 120, grid=grid(2048, 120), stream=stream0)
        del arg122_1
        del arg123_1
        del arg458_1
        del arg459_1
        del buf215
        # Source Nodes: [x_348], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg278_1
        buf218 = buf217; del buf217  # reuse
        buf219 = buf214; del buf214  # reuse
        # Source Nodes: [x_349, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf218, arg460_1, arg461_1, arg124_1, arg125_1, buf219, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg124_1
        del arg125_1
        del arg460_1
        del arg461_1
        del buf218
        # Source Nodes: [x_352, x_353], Original ATen: [aten.convolution, aten.hardswish]
        buf220 = extern_kernels.convolution(buf219, arg279_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf220, (8, 360, 16, 16), (92160, 256, 16, 1))
        del arg279_1
        buf221 = buf220; del buf220  # reuse
        buf222 = reinterpret_tensor(buf213, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf213  # reuse
        buf223 = reinterpret_tensor(buf222, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf222  # reuse
        # Source Nodes: [x_354, x_357, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_23.run(buf221, buf223, arg462_1, arg463_1, arg126_1, arg127_1, 2880, 256, grid=grid(2880), stream=stream0)
        del arg126_1
        del arg127_1
        del arg462_1
        del arg463_1
        # Source Nodes: [x_357, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf224 = extern_kernels.convolution(buf223, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg280_1
        del buf223
        buf225 = reinterpret_tensor(buf224, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf224  # reuse
        # Source Nodes: [x_357, x_se_40, x_se_41, x_se_42], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_27.run(buf225, arg281_1, 256, grid=grid(256), stream=stream0)
        del arg281_1
        # Source Nodes: [x_357, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf226 = extern_kernels.convolution(buf225, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg282_1
        del buf225
        buf227 = buf219; del buf219  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____5___se_gate, x_357, x_358, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_25.run(buf221, buf226, arg283_1, buf227, 2880, 256, grid=grid(2880, 256), stream=stream0)
        del arg283_1
        del buf221
        del buf226
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____5___se_gate, x_357, x_358, x_359, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf228 = extern_kernels.convolution(buf227, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 120, 16, 16), (30720, 256, 16, 1))
        del arg284_1
        del buf227
        buf229 = buf216; del buf216  # reuse
        # Source Nodes: [shortcut_22, x_360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_28.run(buf229, buf228, arg464_1, arg465_1, arg128_1, arg129_1, 2048, 120, grid=grid(2048, 120), stream=stream0)
        del arg128_1
        del arg129_1
        del arg464_1
        del arg465_1
        del buf228
        # Source Nodes: [shortcut_22, x_360, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 720, 16, 16), (184320, 256, 16, 1))
        del arg285_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        buf232 = empty_strided((8, 720, 16, 16), (184320, 1, 11520, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_366, x_369], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_29.run(buf231, arg466_1, arg467_1, arg130_1, arg131_1, buf232, 5760, 256, grid=grid(5760, 256), stream=stream0)
        del arg130_1
        del arg131_1
        del arg466_1
        del arg467_1
        del buf231
        # Source Nodes: [x_369, x_370], Original ATen: [aten.convolution, aten.hardswish]
        buf233 = extern_kernels.convolution(buf232, arg286_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=720, bias=None)
        assert_size_stride(buf233, (8, 720, 8, 8), (46080, 64, 8, 1))
        del arg286_1
        del buf232
        buf234 = buf233; del buf233  # reuse
        buf235 = empty_strided((8, 720, 1, 1), (720, 1, 5760, 5760), device='cuda', dtype=torch.float32)
        buf236 = reinterpret_tensor(buf235, (8, 720, 1, 1), (720, 1, 720, 720), 0); del buf235  # reuse
        # Source Nodes: [x_371, x_374, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_30.run(buf234, buf236, arg468_1, arg469_1, arg132_1, arg133_1, 5760, 64, grid=grid(5760), stream=stream0)
        del arg132_1
        del arg133_1
        del arg468_1
        del arg469_1
        # Source Nodes: [x_374, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf237 = extern_kernels.convolution(buf236, arg287_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg287_1
        del buf236
        buf238 = reinterpret_tensor(buf237, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf237  # reuse
        # Source Nodes: [x_374, x_se_44, x_se_45, x_se_46], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_27.run(buf238, arg288_1, 256, grid=grid(256), stream=stream0)
        del arg288_1
        # Source Nodes: [x_374, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf239 = extern_kernels.convolution(buf238, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 720, 1, 1), (720, 1, 1, 1))
        del arg289_1
        del buf238
        buf240 = empty_strided((8, 720, 8, 8), (46080, 1, 5760, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_374, x_375, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31.run(buf234, buf239, arg290_1, buf240, 5760, 64, grid=grid(5760, 64), stream=stream0)
        del arg290_1
        del buf234
        del buf239
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_374, x_375, x_376, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf241 = extern_kernels.convolution(buf240, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 184, 8, 8), (11776, 64, 8, 1))
        del arg291_1
        del buf240
        buf242 = empty_strided((8, 184, 8, 8), (11776, 1, 1472, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf241, arg470_1, arg471_1, arg134_1, arg135_1, buf242, 1472, 64, grid=grid(1472, 64), stream=stream0)
        del arg134_1
        del arg135_1
        del arg470_1
        del arg471_1
        del buf241
        # Source Nodes: [x_381], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg292_1
        buf244 = buf243; del buf243  # reuse
        buf245 = empty_strided((8, 736, 8, 8), (47104, 1, 5888, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382, x_385], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33.run(buf244, arg472_1, arg473_1, arg136_1, arg137_1, buf245, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg136_1
        del arg137_1
        del arg472_1
        del arg473_1
        del buf244
        # Source Nodes: [x_385, x_386], Original ATen: [aten.convolution, aten.hardswish]
        buf246 = extern_kernels.convolution(buf245, arg293_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf246, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg293_1
        buf247 = buf246; del buf246  # reuse
        buf248 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cuda', dtype=torch.float32)
        buf249 = reinterpret_tensor(buf248, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf248  # reuse
        # Source Nodes: [x_387, x_390, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_34.run(buf247, buf249, arg474_1, arg475_1, arg138_1, arg139_1, 5888, 64, grid=grid(5888), stream=stream0)
        del arg138_1
        del arg139_1
        del arg474_1
        del arg475_1
        # Source Nodes: [x_390, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf250 = extern_kernels.convolution(buf249, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg294_1
        del buf249
        buf251 = reinterpret_tensor(buf250, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf250  # reuse
        # Source Nodes: [x_390, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_35.run(buf251, arg295_1, 384, grid=grid(384), stream=stream0)
        del arg295_1
        # Source Nodes: [x_390, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf252 = extern_kernels.convolution(buf251, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg296_1
        del buf251
        buf253 = buf245; del buf245  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_390, x_391, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_36.run(buf247, buf252, arg297_1, buf253, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg297_1
        del buf247
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_390, x_391, x_392, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf254 = extern_kernels.convolution(buf253, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 184, 8, 8), (11776, 64, 8, 1))
        del arg298_1
        buf255 = buf242; del buf242  # reuse
        # Source Nodes: [shortcut_24, x_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf255, buf254, arg476_1, arg477_1, arg140_1, arg141_1, 512, 184, grid=grid(512, 184), stream=stream0)
        del arg140_1
        del arg141_1
        del arg476_1
        del arg477_1
        del buf254
        # Source Nodes: [x_398], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, arg299_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg299_1
        buf257 = buf256; del buf256  # reuse
        buf258 = buf253; del buf253  # reuse
        # Source Nodes: [x_399, x_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33.run(buf257, arg478_1, arg479_1, arg142_1, arg143_1, buf258, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg142_1
        del arg143_1
        del arg478_1
        del arg479_1
        del buf257
        # Source Nodes: [x_402, x_403], Original ATen: [aten.convolution, aten.hardswish]
        buf259 = extern_kernels.convolution(buf258, arg300_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf259, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg300_1
        buf260 = buf259; del buf259  # reuse
        buf261 = reinterpret_tensor(buf252, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf252  # reuse
        buf262 = reinterpret_tensor(buf261, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf261  # reuse
        # Source Nodes: [x_404, x_407, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_34.run(buf260, buf262, arg480_1, arg481_1, arg144_1, arg145_1, 5888, 64, grid=grid(5888), stream=stream0)
        del arg144_1
        del arg145_1
        del arg480_1
        del arg481_1
        # Source Nodes: [x_407, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf263 = extern_kernels.convolution(buf262, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg301_1
        del buf262
        buf264 = reinterpret_tensor(buf263, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf263  # reuse
        # Source Nodes: [x_407, x_se_52, x_se_53, x_se_54], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_35.run(buf264, arg302_1, 384, grid=grid(384), stream=stream0)
        del arg302_1
        # Source Nodes: [x_407, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf265 = extern_kernels.convolution(buf264, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg303_1
        del buf264
        buf266 = buf258; del buf258  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_407, x_408, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_36.run(buf260, buf265, arg304_1, buf266, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg304_1
        del buf260
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_407, x_408, x_409, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf267 = extern_kernels.convolution(buf266, arg305_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 184, 8, 8), (11776, 64, 8, 1))
        del arg305_1
        buf268 = buf255; del buf255  # reuse
        # Source Nodes: [shortcut_25, x_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf268, buf267, arg482_1, arg483_1, arg146_1, arg147_1, 512, 184, grid=grid(512, 184), stream=stream0)
        del arg146_1
        del arg147_1
        del arg482_1
        del arg483_1
        del buf267
        # Source Nodes: [x_415], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg306_1
        buf270 = buf269; del buf269  # reuse
        buf271 = buf266; del buf266  # reuse
        # Source Nodes: [x_416, x_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33.run(buf270, arg484_1, arg485_1, arg148_1, arg149_1, buf271, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg148_1
        del arg149_1
        del arg484_1
        del arg485_1
        del buf270
        # Source Nodes: [x_419, x_420], Original ATen: [aten.convolution, aten.hardswish]
        buf272 = extern_kernels.convolution(buf271, arg307_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf272, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg307_1
        buf273 = buf272; del buf272  # reuse
        buf274 = reinterpret_tensor(buf265, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf265  # reuse
        buf275 = reinterpret_tensor(buf274, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf274  # reuse
        # Source Nodes: [x_421, x_424, x_se_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_34.run(buf273, buf275, arg486_1, arg487_1, arg150_1, arg151_1, 5888, 64, grid=grid(5888), stream=stream0)
        del arg150_1
        del arg151_1
        del arg486_1
        del arg487_1
        # Source Nodes: [x_424, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf276 = extern_kernels.convolution(buf275, arg308_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg308_1
        del buf275
        buf277 = reinterpret_tensor(buf276, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf276  # reuse
        # Source Nodes: [x_424, x_se_56, x_se_57, x_se_58], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_35.run(buf277, arg309_1, 384, grid=grid(384), stream=stream0)
        del arg309_1
        # Source Nodes: [x_424, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf278 = extern_kernels.convolution(buf277, arg310_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg310_1
        del buf277
        buf279 = buf271; del buf271  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_424, x_425, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_36.run(buf273, buf278, arg311_1, buf279, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg311_1
        del buf273
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_424, x_425, x_426, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf280 = extern_kernels.convolution(buf279, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 184, 8, 8), (11776, 64, 8, 1))
        del arg312_1
        buf281 = buf268; del buf268  # reuse
        # Source Nodes: [shortcut_26, x_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf281, buf280, arg488_1, arg489_1, arg152_1, arg153_1, 512, 184, grid=grid(512, 184), stream=stream0)
        del arg152_1
        del arg153_1
        del arg488_1
        del arg489_1
        del buf280
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, arg313_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg313_1
        buf283 = buf282; del buf282  # reuse
        buf284 = buf279; del buf279  # reuse
        # Source Nodes: [x_433, x_436], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33.run(buf283, arg490_1, arg491_1, arg154_1, arg155_1, buf284, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg154_1
        del arg155_1
        del arg490_1
        del arg491_1
        del buf283
        # Source Nodes: [x_436, x_437], Original ATen: [aten.convolution, aten.hardswish]
        buf285 = extern_kernels.convolution(buf284, arg314_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf285, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg314_1
        buf286 = buf285; del buf285  # reuse
        buf287 = reinterpret_tensor(buf278, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf278  # reuse
        buf288 = reinterpret_tensor(buf287, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf287  # reuse
        # Source Nodes: [x_438, x_441, x_se_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_34.run(buf286, buf288, arg492_1, arg493_1, arg156_1, arg157_1, 5888, 64, grid=grid(5888), stream=stream0)
        del arg156_1
        del arg157_1
        del arg492_1
        del arg493_1
        # Source Nodes: [x_441, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf289 = extern_kernels.convolution(buf288, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg315_1
        del buf288
        buf290 = reinterpret_tensor(buf289, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf289  # reuse
        # Source Nodes: [x_441, x_se_60, x_se_61, x_se_62], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_35.run(buf290, arg316_1, 384, grid=grid(384), stream=stream0)
        del arg316_1
        # Source Nodes: [x_441, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf291 = extern_kernels.convolution(buf290, arg317_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg317_1
        del buf290
        buf292 = buf284; del buf284  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_441, x_442, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_36.run(buf286, buf291, arg318_1, buf292, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg318_1
        del buf286
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_441, x_442, x_443, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf293 = extern_kernels.convolution(buf292, arg319_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 184, 8, 8), (11776, 64, 8, 1))
        del arg319_1
        buf294 = buf281; del buf281  # reuse
        # Source Nodes: [shortcut_27, x_444], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf294, buf293, arg494_1, arg495_1, arg158_1, arg159_1, 512, 184, grid=grid(512, 184), stream=stream0)
        del arg158_1
        del arg159_1
        del arg494_1
        del arg495_1
        del buf293
        # Source Nodes: [x_449], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, arg320_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg320_1
        buf296 = buf295; del buf295  # reuse
        buf297 = buf292; del buf292  # reuse
        # Source Nodes: [x_450, x_453], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_33.run(buf296, arg496_1, arg497_1, arg160_1, arg161_1, buf297, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg160_1
        del arg161_1
        del arg496_1
        del arg497_1
        del buf296
        # Source Nodes: [x_453, x_454], Original ATen: [aten.convolution, aten.hardswish]
        buf298 = extern_kernels.convolution(buf297, arg321_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf298, (8, 736, 8, 8), (47104, 64, 8, 1))
        del arg321_1
        buf299 = buf298; del buf298  # reuse
        buf300 = reinterpret_tensor(buf291, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf291  # reuse
        buf301 = reinterpret_tensor(buf300, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf300  # reuse
        # Source Nodes: [x_455, x_458, x_se_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_34.run(buf299, buf301, arg498_1, arg499_1, arg162_1, arg163_1, 5888, 64, grid=grid(5888), stream=stream0)
        del arg162_1
        del arg163_1
        del arg498_1
        del arg499_1
        # Source Nodes: [x_458, x_se_64, x_se_65], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf302 = extern_kernels.convolution(buf301, arg322_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg322_1
        del buf301
        buf303 = reinterpret_tensor(buf302, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf302  # reuse
        # Source Nodes: [x_458, x_se_64, x_se_65, x_se_66], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_35.run(buf303, arg323_1, 384, grid=grid(384), stream=stream0)
        del arg323_1
        # Source Nodes: [x_458, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf304 = extern_kernels.convolution(buf303, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg324_1
        del buf303
        buf305 = buf297; del buf297  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____5___se_gate, x_458, x_459, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_36.run(buf299, buf304, arg325_1, buf305, 5888, 64, grid=grid(5888, 64), stream=stream0)
        del arg325_1
        del buf299
        del buf304
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____5___se_gate, x_458, x_459, x_460, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf306 = extern_kernels.convolution(buf305, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 184, 8, 8), (11776, 64, 8, 1))
        del arg326_1
        del buf305
        buf307 = buf294; del buf294  # reuse
        # Source Nodes: [shortcut_28, x_461], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf307, buf306, arg500_1, arg501_1, arg164_1, arg165_1, 512, 184, grid=grid(512, 184), stream=stream0)
        del arg164_1
        del arg165_1
        del arg500_1
        del arg501_1
        del buf306
        # Source Nodes: [shortcut_28, x_461, x_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf308 = extern_kernels.convolution(buf307, arg327_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (8, 1104, 8, 8), (70656, 64, 8, 1))
        del arg327_1
        del buf307
        buf309 = buf308; del buf308  # reuse
        buf310 = empty_strided((8, 1104, 8, 8), (70656, 1, 8832, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_467, x_470], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf309, arg502_1, arg503_1, arg166_1, arg167_1, buf310, 8832, 64, grid=grid(8832, 64), stream=stream0)
        del arg166_1
        del arg167_1
        del arg502_1
        del arg503_1
        del buf309
        # Source Nodes: [x_470, x_471], Original ATen: [aten.convolution, aten.hardswish]
        buf311 = extern_kernels.convolution(buf310, arg328_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf311, (8, 1104, 8, 8), (70656, 64, 8, 1))
        del arg328_1
        buf312 = buf311; del buf311  # reuse
        buf313 = empty_strided((8, 1104, 1, 1), (1104, 1, 8832, 8832), device='cuda', dtype=torch.float32)
        buf314 = reinterpret_tensor(buf313, (8, 1104, 1, 1), (1104, 1, 1104, 1104), 0); del buf313  # reuse
        # Source Nodes: [x_472, x_475, x_se_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39.run(buf312, buf314, arg504_1, arg505_1, arg168_1, arg169_1, 8832, 64, grid=grid(8832), stream=stream0)
        del arg168_1
        del arg169_1
        del arg504_1
        del arg505_1
        # Source Nodes: [x_475, x_se_68, x_se_69], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf315 = extern_kernels.convolution(buf314, arg329_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg329_1
        del buf314
        buf316 = reinterpret_tensor(buf315, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf315  # reuse
        # Source Nodes: [x_475, x_se_68, x_se_69, x_se_70], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_35.run(buf316, arg330_1, 384, grid=grid(384), stream=stream0)
        del arg330_1
        # Source Nodes: [x_475, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf317 = extern_kernels.convolution(buf316, arg331_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (8, 1104, 1, 1), (1104, 1, 1, 1))
        del arg331_1
        del buf316
        buf318 = buf310; del buf310  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____6___se_gate, x_475, x_476, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_40.run(buf312, buf317, arg332_1, buf318, 8832, 64, grid=grid(8832, 64), stream=stream0)
        del arg332_1
        del buf312
        del buf317
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____6___se_gate, x_475, x_476, x_477, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul]
        buf319 = extern_kernels.convolution(buf318, arg333_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 224, 8, 8), (14336, 64, 8, 1))
        del arg333_1
        del buf318
        buf320 = empty_strided((8, 224, 8, 8), (14336, 1, 1792, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf319, arg506_1, arg507_1, arg170_1, arg171_1, buf320, 1792, 64, grid=grid(1792, 64), stream=stream0)
        del arg170_1
        del arg171_1
        del arg506_1
        del arg507_1
        del buf319
        # Source Nodes: [x_478, x_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf321 = extern_kernels.convolution(buf320, arg334_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (8, 1344, 8, 8), (86016, 64, 8, 1))
        del arg334_1
        del buf320
        buf322 = buf321; del buf321  # reuse
        buf323 = empty_strided((8, 1344, 1, 1), (1344, 1, 10752, 10752), device='cuda', dtype=torch.float32)
        buf324 = reinterpret_tensor(buf323, (8, 1344, 1, 1), (1344, 1, 1344, 1344), 0); del buf323  # reuse
        # Source Nodes: [x_483, x_488, x_489], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_42.run(buf322, buf324, arg508_1, arg509_1, arg172_1, arg173_1, 10752, 64, grid=grid(10752), stream=stream0)
        del arg172_1
        del arg173_1
        del arg508_1
        del arg509_1
        del buf322
        # Source Nodes: [x_488, x_489, x_492], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf325 = extern_kernels.convolution(buf324, arg335_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (8, 1984, 1, 1), (1984, 1, 1, 1))
        del arg335_1
        del buf324
        buf326 = buf325; del buf325  # reuse
        # Source Nodes: [x_493], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_43.run(buf326, 15872, grid=grid(15872), stream=stream0)
        buf327 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_495], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg175_1, reinterpret_tensor(buf326, (8, 1984), (1984, 1), 0), reinterpret_tensor(arg174_1, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf327)
        del arg174_1
        del arg175_1
        return (buf327, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1000, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((120, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((8, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((120, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((200, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((200, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((72, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((360, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((360, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((24, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((360, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((720, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((720, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((32, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((720, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((184, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((48, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1104, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((224, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1984, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetv3_b', benchmark_compiled_module)
