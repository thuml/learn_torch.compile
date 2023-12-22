
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3mdeneejappqqe5fk6zuhypf6spx7mpv4obazedrrc25iia6ca.py
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
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
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


# kernel path: /tmp/torchinductor_youkaichao/wg/cwghnqoh7lyetnyjgv4eofx76xvisklwosmuukbl3hbu432p7hne.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# shortcut => relu
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzcjqwuhpxxtjnw7xo64qnot2x7mwa2sbrpnd42os3dgiz3bqjq.py
# Source Nodes: [shortcut_1, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_1 => add_8
# x_17 => add_7, mul_10, mul_11, sub_3
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
    ynumel = 100352
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (12544*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/lq/clq34wicz5a3g4cs4srczjrekxgfv3vlb3mcgrovfi7pber6misv.py
# Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_23 => add_10, mul_13, mul_14, sub_4
# x_26 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmc2kuwyof2k2ow54hlyhmxpwqqaw2infwxyq7ygbr7unwlinz4i.py
# Source Nodes: [x_28, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_28 => add_12, mul_16, mul_17, sub_5
# x_31 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chpmjnwjmjpegbxh3glj3wmtsec2szi72u56rzubuvekllfgzcld.py
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_34 => add_14, mul_19, mul_20, sub_6
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
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46f7wemeh5b47es7bwqn5xfqk345irsqzblnwwjj35xtjk44s4z.py
# Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_39 => add_16, mul_22, mul_23, sub_7
# x_42 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3x/c3xt7zbpq736shncr757jyjq53w7orp4tuyoxpixxj7rtqt7x724.py
# Source Nodes: [shortcut_3, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_21
# x_50 => add_20, mul_28, mul_29, sub_9
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
    ynumel = 25088
    xnumel = 24
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (24*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bi/cbivdrel2gujdppimpkzz4iln2hg64a3cmsytzzazkqsihr35ydq.py
# Source Nodes: [x_73, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_73 => add_30, mul_40, mul_41, sub_13
# x_76 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/clygybetpnurmp7ejapt3h5z4trpkj3y6ht63pityx3jdy7pifsc.py
# Source Nodes: [x_78, x_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_78 => add_32, mul_43, mul_44, sub_14
# x_81 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (144*x2) + (112896*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4sg64z7u2rjadihoxr3qj5x6odhpsif7dx4izh7ev3rohm76mj.py
# Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_84 => add_34, mul_46, mul_47, sub_15
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (32*x2) + (25088*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qfoc5w4qgnlvjts5mztks35zkpzm5hzy3vpiu5zyo5fi2mrtez.py
# Source Nodes: [x_89, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_89 => add_36, mul_49, mul_50, sub_16
# x_92 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (96*x2) + (75264*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkb74vy462wslpki62ypx4zumqyn24qy5ingzy2jw6hx7lhijsd.py
# Source Nodes: [shortcut_6, x_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_6 => add_41
# x_100 => add_40, mul_55, mul_56, sub_18
triton_poi_fused__native_batch_norm_legit_no_training_add_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2 + (32*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr67xnajm2i2vhg2s2lcmjg7enwpqoq7k6sypyjayg42pswbpqs6.py
# Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_106 => add_43, mul_58, mul_59, sub_19
# x_109 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkqstke7gxposvw7cmaol2phni65675uhpopfuflhsh2vtp6lvw.py
# Source Nodes: [x_145, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_145 => add_59, mul_79, mul_80, sub_26
# x_148 => relu_18
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (192*x2) + (37632*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdt74avg4kei5ho2wdm32y2gbefzmf7podogpgcz267twe6mcg3f.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_151 => add_61, mul_82, mul_83, sub_27
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (64*x2) + (12544*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4uztgypdpjn34oktjq4ceq3ngi3gineplkmfwhdcs67l7eljm7u.py
# Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_10 => add_68
# x_167 => add_67, mul_91, mul_92, sub_30
triton_poi_fused__native_batch_norm_legit_no_training_add_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/circvvipmmtzxdksjm2p6iccv33ysza55rsbwketgxxxfnt7vr7n.py
# Source Nodes: [x_173, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_173 => add_70, mul_94, mul_95, sub_31
# x_176 => relu_21
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjf6buedqkmcdgo7lna4sk5fqoysmps2jcfmvxh276d433syh7d.py
# Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_218 => add_88, mul_118, mul_119, sub_39
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5q3lraqnvgutagl7hdv3jjagpnnv7mbslsc6cb4bpq4wzrlpect.py
# Source Nodes: [x_223, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_223 => add_90, mul_121, mul_122, sub_40
# x_226 => relu_27
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csobf4y2hh4wzsepisn3blxvu5bajv26qgf3mfr7xsjb7w62m5ta.py
# Source Nodes: [shortcut_14, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_14 => add_95
# x_234 => add_94, mul_127, mul_128, sub_42
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2 + (112*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3e6jy2ozkbjjpv6jbhyib5tdxfklvebxvqh6p3ep6gqn6ddtvtz.py
# Source Nodes: [x_257, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_257 => add_104, mul_139, mul_140, sub_46
# x_260 => relu_31
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (336*x2) + (65856*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbak3ybdv3z4bhpzy4kahkxuhmc7cbzgchzye2buxkhegujf6uk.py
# Source Nodes: [x_279, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_279 => add_113, mul_151, mul_152, sub_50
# x_282 => relu_34
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbtm6c7ahw64cwcmsukneceex3tsqtiwaopvj4x4vr66tmllcq5.py
# Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_285 => add_115, mul_154, mul_155, sub_51
triton_poi_fused__native_batch_norm_legit_no_training_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (184*x2) + (9016*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/crerdkw7cbuzyamgq7zg4xdipcktjr72oo3phmaqtzswntpv7j7z.py
# Source Nodes: [x_290, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_290 => add_117, mul_157, mul_158, sub_52
# x_293 => relu_35
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8832
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (1104*x2) + (54096*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cij5ufjzrzkvrvb4bmeipyj5b5gu6taijp6yhkm34jmtvyddpgz4.py
# Source Nodes: [shortcut_18, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_18 => add_122
# x_301 => add_121, mul_163, mul_164, sub_54
triton_poi_fused__native_batch_norm_legit_no_training_add_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 184
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (9016*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (184*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/xk/cxk4f55mwice376ptbayyjcraiismorzl275kjesjacpmfs4qqcf.py
# Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_352 => add_142, mul_190, mul_191, sub_63
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2816
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 352
    y1 = (yindex // 352)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (352*x2) + (17248*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy56wgo5qeoo6a7hdd64e2i7pu5ypfcgox5dec7yjkwsmlidfket.py
# Source Nodes: [x_358, x_362, x_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_358 => add_144, mul_193, mul_194, sub_64
# x_362 => relu_43
# x_363 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1984
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 49.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, ), (1, ))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (24, ), (1, ))
    assert_size_stride(arg13_1, (24, ), (1, ))
    assert_size_stride(arg14_1, (24, ), (1, ))
    assert_size_stride(arg15_1, (24, ), (1, ))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (24, ), (1, ))
    assert_size_stride(arg19_1, (24, ), (1, ))
    assert_size_stride(arg20_1, (24, ), (1, ))
    assert_size_stride(arg21_1, (24, ), (1, ))
    assert_size_stride(arg22_1, (24, ), (1, ))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (24, ), (1, ))
    assert_size_stride(arg26_1, (144, ), (1, ))
    assert_size_stride(arg27_1, (144, ), (1, ))
    assert_size_stride(arg28_1, (144, ), (1, ))
    assert_size_stride(arg29_1, (144, ), (1, ))
    assert_size_stride(arg30_1, (32, ), (1, ))
    assert_size_stride(arg31_1, (32, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (32, ), (1, ))
    assert_size_stride(arg37_1, (32, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, ), (1, ))
    assert_size_stride(arg42_1, (32, ), (1, ))
    assert_size_stride(arg43_1, (32, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (32, ), (1, ))
    assert_size_stride(arg49_1, (32, ), (1, ))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (192, ), (1, ))
    assert_size_stride(arg53_1, (192, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (192, ), (1, ))
    assert_size_stride(arg57_1, (192, ), (1, ))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (64, ), (1, ))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (64, ), (1, ))
    assert_size_stride(arg67_1, (64, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (64, ), (1, ))
    assert_size_stride(arg73_1, (64, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (112, ), (1, ))
    assert_size_stride(arg79_1, (112, ), (1, ))
    assert_size_stride(arg80_1, (672, ), (1, ))
    assert_size_stride(arg81_1, (672, ), (1, ))
    assert_size_stride(arg82_1, (672, ), (1, ))
    assert_size_stride(arg83_1, (672, ), (1, ))
    assert_size_stride(arg84_1, (112, ), (1, ))
    assert_size_stride(arg85_1, (112, ), (1, ))
    assert_size_stride(arg86_1, (672, ), (1, ))
    assert_size_stride(arg87_1, (672, ), (1, ))
    assert_size_stride(arg88_1, (672, ), (1, ))
    assert_size_stride(arg89_1, (672, ), (1, ))
    assert_size_stride(arg90_1, (112, ), (1, ))
    assert_size_stride(arg91_1, (112, ), (1, ))
    assert_size_stride(arg92_1, (336, ), (1, ))
    assert_size_stride(arg93_1, (336, ), (1, ))
    assert_size_stride(arg94_1, (336, ), (1, ))
    assert_size_stride(arg95_1, (336, ), (1, ))
    assert_size_stride(arg96_1, (112, ), (1, ))
    assert_size_stride(arg97_1, (112, ), (1, ))
    assert_size_stride(arg98_1, (672, ), (1, ))
    assert_size_stride(arg99_1, (672, ), (1, ))
    assert_size_stride(arg100_1, (672, ), (1, ))
    assert_size_stride(arg101_1, (672, ), (1, ))
    assert_size_stride(arg102_1, (184, ), (1, ))
    assert_size_stride(arg103_1, (184, ), (1, ))
    assert_size_stride(arg104_1, (1104, ), (1, ))
    assert_size_stride(arg105_1, (1104, ), (1, ))
    assert_size_stride(arg106_1, (1104, ), (1, ))
    assert_size_stride(arg107_1, (1104, ), (1, ))
    assert_size_stride(arg108_1, (184, ), (1, ))
    assert_size_stride(arg109_1, (184, ), (1, ))
    assert_size_stride(arg110_1, (1104, ), (1, ))
    assert_size_stride(arg111_1, (1104, ), (1, ))
    assert_size_stride(arg112_1, (1104, ), (1, ))
    assert_size_stride(arg113_1, (1104, ), (1, ))
    assert_size_stride(arg114_1, (184, ), (1, ))
    assert_size_stride(arg115_1, (184, ), (1, ))
    assert_size_stride(arg116_1, (1104, ), (1, ))
    assert_size_stride(arg117_1, (1104, ), (1, ))
    assert_size_stride(arg118_1, (1104, ), (1, ))
    assert_size_stride(arg119_1, (1104, ), (1, ))
    assert_size_stride(arg120_1, (184, ), (1, ))
    assert_size_stride(arg121_1, (184, ), (1, ))
    assert_size_stride(arg122_1, (1104, ), (1, ))
    assert_size_stride(arg123_1, (1104, ), (1, ))
    assert_size_stride(arg124_1, (1104, ), (1, ))
    assert_size_stride(arg125_1, (1104, ), (1, ))
    assert_size_stride(arg126_1, (352, ), (1, ))
    assert_size_stride(arg127_1, (352, ), (1, ))
    assert_size_stride(arg128_1, (1984, ), (1, ))
    assert_size_stride(arg129_1, (1984, ), (1, ))
    assert_size_stride(arg130_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg131_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg132_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg133_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg134_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg135_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg137_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg138_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg139_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg140_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg141_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg143_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg144_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg145_1, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg146_1, (96, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg147_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg148_1, (32, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg149_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg150_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg151_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg152_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg153_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg155_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg156_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg157_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg158_1, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg159_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg160_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg161_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg162_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg163_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg164_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg165_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg166_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg167_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg168_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg169_1, (112, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg170_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg171_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg172_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg173_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg174_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg175_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg176_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg177_1, (336, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg178_1, (112, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg179_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg180_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg181_1, (184, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg182_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg183_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg184_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg185_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg186_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg187_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg188_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg189_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg190_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg191_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg192_1, (1104, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg193_1, (352, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg194_1, (1984, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(arg195_1, (1000, 1984), (1984, 1))
    assert_size_stride(arg196_1, (1000, ), (1, ))
    assert_size_stride(arg197_1, (16, ), (1, ))
    assert_size_stride(arg198_1, (16, ), (1, ))
    assert_size_stride(arg199_1, (16, ), (1, ))
    assert_size_stride(arg200_1, (16, ), (1, ))
    assert_size_stride(arg201_1, (16, ), (1, ))
    assert_size_stride(arg202_1, (16, ), (1, ))
    assert_size_stride(arg203_1, (16, ), (1, ))
    assert_size_stride(arg204_1, (16, ), (1, ))
    assert_size_stride(arg205_1, (96, ), (1, ))
    assert_size_stride(arg206_1, (96, ), (1, ))
    assert_size_stride(arg207_1, (96, ), (1, ))
    assert_size_stride(arg208_1, (96, ), (1, ))
    assert_size_stride(arg209_1, (24, ), (1, ))
    assert_size_stride(arg210_1, (24, ), (1, ))
    assert_size_stride(arg211_1, (24, ), (1, ))
    assert_size_stride(arg212_1, (24, ), (1, ))
    assert_size_stride(arg213_1, (24, ), (1, ))
    assert_size_stride(arg214_1, (24, ), (1, ))
    assert_size_stride(arg215_1, (24, ), (1, ))
    assert_size_stride(arg216_1, (24, ), (1, ))
    assert_size_stride(arg217_1, (24, ), (1, ))
    assert_size_stride(arg218_1, (24, ), (1, ))
    assert_size_stride(arg219_1, (24, ), (1, ))
    assert_size_stride(arg220_1, (24, ), (1, ))
    assert_size_stride(arg221_1, (24, ), (1, ))
    assert_size_stride(arg222_1, (24, ), (1, ))
    assert_size_stride(arg223_1, (144, ), (1, ))
    assert_size_stride(arg224_1, (144, ), (1, ))
    assert_size_stride(arg225_1, (144, ), (1, ))
    assert_size_stride(arg226_1, (144, ), (1, ))
    assert_size_stride(arg227_1, (32, ), (1, ))
    assert_size_stride(arg228_1, (32, ), (1, ))
    assert_size_stride(arg229_1, (96, ), (1, ))
    assert_size_stride(arg230_1, (96, ), (1, ))
    assert_size_stride(arg231_1, (96, ), (1, ))
    assert_size_stride(arg232_1, (96, ), (1, ))
    assert_size_stride(arg233_1, (32, ), (1, ))
    assert_size_stride(arg234_1, (32, ), (1, ))
    assert_size_stride(arg235_1, (192, ), (1, ))
    assert_size_stride(arg236_1, (192, ), (1, ))
    assert_size_stride(arg237_1, (192, ), (1, ))
    assert_size_stride(arg238_1, (192, ), (1, ))
    assert_size_stride(arg239_1, (32, ), (1, ))
    assert_size_stride(arg240_1, (32, ), (1, ))
    assert_size_stride(arg241_1, (192, ), (1, ))
    assert_size_stride(arg242_1, (192, ), (1, ))
    assert_size_stride(arg243_1, (192, ), (1, ))
    assert_size_stride(arg244_1, (192, ), (1, ))
    assert_size_stride(arg245_1, (32, ), (1, ))
    assert_size_stride(arg246_1, (32, ), (1, ))
    assert_size_stride(arg247_1, (192, ), (1, ))
    assert_size_stride(arg248_1, (192, ), (1, ))
    assert_size_stride(arg249_1, (192, ), (1, ))
    assert_size_stride(arg250_1, (192, ), (1, ))
    assert_size_stride(arg251_1, (64, ), (1, ))
    assert_size_stride(arg252_1, (64, ), (1, ))
    assert_size_stride(arg253_1, (192, ), (1, ))
    assert_size_stride(arg254_1, (192, ), (1, ))
    assert_size_stride(arg255_1, (192, ), (1, ))
    assert_size_stride(arg256_1, (192, ), (1, ))
    assert_size_stride(arg257_1, (64, ), (1, ))
    assert_size_stride(arg258_1, (64, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (64, ), (1, ))
    assert_size_stride(arg264_1, (64, ), (1, ))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (64, ), (1, ))
    assert_size_stride(arg270_1, (64, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (384, ), (1, ))
    assert_size_stride(arg275_1, (112, ), (1, ))
    assert_size_stride(arg276_1, (112, ), (1, ))
    assert_size_stride(arg277_1, (672, ), (1, ))
    assert_size_stride(arg278_1, (672, ), (1, ))
    assert_size_stride(arg279_1, (672, ), (1, ))
    assert_size_stride(arg280_1, (672, ), (1, ))
    assert_size_stride(arg281_1, (112, ), (1, ))
    assert_size_stride(arg282_1, (112, ), (1, ))
    assert_size_stride(arg283_1, (672, ), (1, ))
    assert_size_stride(arg284_1, (672, ), (1, ))
    assert_size_stride(arg285_1, (672, ), (1, ))
    assert_size_stride(arg286_1, (672, ), (1, ))
    assert_size_stride(arg287_1, (112, ), (1, ))
    assert_size_stride(arg288_1, (112, ), (1, ))
    assert_size_stride(arg289_1, (336, ), (1, ))
    assert_size_stride(arg290_1, (336, ), (1, ))
    assert_size_stride(arg291_1, (336, ), (1, ))
    assert_size_stride(arg292_1, (336, ), (1, ))
    assert_size_stride(arg293_1, (112, ), (1, ))
    assert_size_stride(arg294_1, (112, ), (1, ))
    assert_size_stride(arg295_1, (672, ), (1, ))
    assert_size_stride(arg296_1, (672, ), (1, ))
    assert_size_stride(arg297_1, (672, ), (1, ))
    assert_size_stride(arg298_1, (672, ), (1, ))
    assert_size_stride(arg299_1, (184, ), (1, ))
    assert_size_stride(arg300_1, (184, ), (1, ))
    assert_size_stride(arg301_1, (1104, ), (1, ))
    assert_size_stride(arg302_1, (1104, ), (1, ))
    assert_size_stride(arg303_1, (1104, ), (1, ))
    assert_size_stride(arg304_1, (1104, ), (1, ))
    assert_size_stride(arg305_1, (184, ), (1, ))
    assert_size_stride(arg306_1, (184, ), (1, ))
    assert_size_stride(arg307_1, (1104, ), (1, ))
    assert_size_stride(arg308_1, (1104, ), (1, ))
    assert_size_stride(arg309_1, (1104, ), (1, ))
    assert_size_stride(arg310_1, (1104, ), (1, ))
    assert_size_stride(arg311_1, (184, ), (1, ))
    assert_size_stride(arg312_1, (184, ), (1, ))
    assert_size_stride(arg313_1, (1104, ), (1, ))
    assert_size_stride(arg314_1, (1104, ), (1, ))
    assert_size_stride(arg315_1, (1104, ), (1, ))
    assert_size_stride(arg316_1, (1104, ), (1, ))
    assert_size_stride(arg317_1, (184, ), (1, ))
    assert_size_stride(arg318_1, (184, ), (1, ))
    assert_size_stride(arg319_1, (1104, ), (1, ))
    assert_size_stride(arg320_1, (1104, ), (1, ))
    assert_size_stride(arg321_1, (1104, ), (1, ))
    assert_size_stride(arg322_1, (1104, ), (1, ))
    assert_size_stride(arg323_1, (352, ), (1, ))
    assert_size_stride(arg324_1, (352, ), (1, ))
    assert_size_stride(arg325_1, (1984, ), (1, ))
    assert_size_stride(arg326_1, (1984, ), (1, ))
    assert_size_stride(arg327_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg327_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg327_1
        buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg130_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg130_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del buf1
        buf3 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg197_1, arg198_1, arg0_1, arg1_1, buf3, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg0_1
        del arg197_1
        del arg198_1
        del arg1_1
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg131_1
        buf5 = reinterpret_tensor(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf2  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, arg199_1, arg200_1, arg2_1, arg3_1, buf5, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg199_1
        del arg200_1
        del arg2_1
        del arg3_1
        del buf4
        # Source Nodes: [x_10, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg132_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf6, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg132_1
        buf7 = buf5; del buf5  # reuse
        # Source Nodes: [x_11, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, arg201_1, arg202_1, arg4_1, arg5_1, buf7, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg201_1
        del arg202_1
        del arg4_1
        del arg5_1
        del buf6
        # Source Nodes: [x_11, x_14, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf7, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg133_1
        del buf7
        buf9 = buf3; del buf3  # reuse
        # Source Nodes: [shortcut_1, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf9, buf8, arg203_1, arg204_1, arg6_1, arg7_1, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del arg203_1
        del arg204_1
        del arg6_1
        del arg7_1
        del buf8
        # Source Nodes: [shortcut_1, x_17, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        del arg134_1
        del buf9
        buf11 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf10, arg205_1, arg206_1, arg8_1, arg9_1, buf11, 768, 12544, grid=grid(768, 12544), stream=stream0)
        del arg205_1
        del arg206_1
        del arg8_1
        del arg9_1
        del buf10
        # Source Nodes: [x_23, x_26, x_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg135_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf12, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg135_1
        del buf11
        buf13 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf12, arg207_1, arg208_1, arg10_1, arg11_1, buf13, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg207_1
        del arg208_1
        del buf12
        # Source Nodes: [x_28, x_31, x_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf13, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg136_1
        del buf13
        buf15 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf14, arg209_1, arg210_1, arg12_1, arg13_1, buf15, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg209_1
        del arg210_1
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg137_1
        buf17 = reinterpret_tensor(buf14, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf14  # reuse
        # Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf16, arg211_1, arg212_1, arg14_1, arg15_1, buf17, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg14_1
        del arg15_1
        del arg211_1
        del arg212_1
        del buf16
        # Source Nodes: [x_39, x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf18, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg138_1
        buf19 = buf17; del buf17  # reuse
        # Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf18, arg213_1, arg214_1, arg16_1, arg17_1, buf19, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg16_1
        del arg17_1
        del arg213_1
        del arg214_1
        del buf18
        # Source Nodes: [x_44, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf20 = extern_kernels.convolution(buf19, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg139_1
        del buf19
        buf21 = buf15; del buf15  # reuse
        # Source Nodes: [shortcut_3, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf21, buf20, arg215_1, arg216_1, arg18_1, arg19_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg18_1
        del arg19_1
        del arg215_1
        del arg216_1
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg140_1
        buf23 = reinterpret_tensor(buf20, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf20  # reuse
        # Source Nodes: [x_56, x_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf22, arg217_1, arg218_1, arg20_1, arg21_1, buf23, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg20_1
        del arg217_1
        del arg218_1
        del arg21_1
        del buf22
        # Source Nodes: [x_56, x_59, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf24, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg141_1
        buf25 = buf23; del buf23  # reuse
        # Source Nodes: [x_61, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf24, arg219_1, arg220_1, arg22_1, arg23_1, buf25, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg219_1
        del arg220_1
        del arg22_1
        del arg23_1
        del buf24
        # Source Nodes: [x_61, x_64, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf26 = extern_kernels.convolution(buf25, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg142_1
        del buf25
        buf27 = buf21; del buf21  # reuse
        # Source Nodes: [shortcut_4, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf27, buf26, arg221_1, arg222_1, arg24_1, arg25_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg221_1
        del arg222_1
        del arg24_1
        del arg25_1
        del buf26
        # Source Nodes: [shortcut_4, x_67, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 144, 56, 56), (451584, 3136, 56, 1))
        del arg143_1
        buf29 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf28, arg223_1, arg224_1, arg26_1, arg27_1, buf29, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del arg223_1
        del arg224_1
        del arg26_1
        del arg27_1
        del buf28
        # Source Nodes: [x_73, x_76, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg144_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf30, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg144_1
        del buf29
        buf31 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78, x_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf30, arg225_1, arg226_1, arg28_1, arg29_1, buf31, 1152, 784, grid=grid(1152, 784), stream=stream0)
        del arg225_1
        del arg226_1
        del arg28_1
        del arg29_1
        del buf30
        # Source Nodes: [x_78, x_81, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf32 = extern_kernels.convolution(buf31, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg145_1
        del buf31
        buf33 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf32, arg227_1, arg228_1, arg30_1, arg31_1, buf33, 256, 784, grid=grid(256, 784), stream=stream0)
        del arg227_1
        del arg228_1
        del arg30_1
        del arg31_1
        del buf32
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 96, 28, 28), (75264, 784, 28, 1))
        del arg146_1
        buf35 = reinterpret_tensor(buf27, (8, 96, 28, 28), (75264, 1, 2688, 96), 0); del buf27  # reuse
        # Source Nodes: [x_89, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf34, arg229_1, arg230_1, arg32_1, arg33_1, buf35, 768, 784, grid=grid(768, 784), stream=stream0)
        del arg229_1
        del arg230_1
        del arg32_1
        del arg33_1
        del buf34
        # Source Nodes: [x_89, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf36 = extern_kernels.convolution(buf35, arg147_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf36, (8, 96, 28, 28), (75264, 784, 28, 1))
        del arg147_1
        buf37 = buf35; del buf35  # reuse
        # Source Nodes: [x_94, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf36, arg231_1, arg232_1, arg34_1, arg35_1, buf37, 768, 784, grid=grid(768, 784), stream=stream0)
        del arg231_1
        del arg232_1
        del arg34_1
        del arg35_1
        del buf36
        # Source Nodes: [x_94, x_97, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf38 = extern_kernels.convolution(buf37, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg148_1
        buf39 = buf33; del buf33  # reuse
        # Source Nodes: [shortcut_6, x_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf39, buf38, arg233_1, arg234_1, arg36_1, arg37_1, 6272, 32, grid=grid(6272, 32), stream=stream0)
        del arg233_1
        del arg234_1
        del arg36_1
        del arg37_1
        del buf38
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg149_1
        buf41 = reinterpret_tensor(buf0, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf0  # reuse
        # Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf40, arg235_1, arg236_1, arg38_1, arg39_1, buf41, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg235_1
        del arg236_1
        del arg38_1
        del arg39_1
        del buf40
        # Source Nodes: [x_106, x_109, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg150_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf42, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg150_1
        buf43 = buf41; del buf41  # reuse
        # Source Nodes: [x_111, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf42, arg237_1, arg238_1, arg40_1, arg41_1, buf43, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg237_1
        del arg238_1
        del arg40_1
        del arg41_1
        del buf42
        # Source Nodes: [x_111, x_114, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf44 = extern_kernels.convolution(buf43, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg151_1
        buf45 = buf39; del buf39  # reuse
        # Source Nodes: [shortcut_7, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf45, buf44, arg239_1, arg240_1, arg42_1, arg43_1, 6272, 32, grid=grid(6272, 32), stream=stream0)
        del arg239_1
        del arg240_1
        del arg42_1
        del arg43_1
        del buf44
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg152_1
        buf47 = buf43; del buf43  # reuse
        # Source Nodes: [x_123, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf46, arg241_1, arg242_1, arg44_1, arg45_1, buf47, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg241_1
        del arg242_1
        del arg44_1
        del arg45_1
        del buf46
        # Source Nodes: [x_123, x_126, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf48 = extern_kernels.convolution(buf47, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf48, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg153_1
        buf49 = buf47; del buf47  # reuse
        # Source Nodes: [x_128, x_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf48, arg243_1, arg244_1, arg46_1, arg47_1, buf49, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg243_1
        del arg244_1
        del arg46_1
        del arg47_1
        del buf48
        # Source Nodes: [x_128, x_131, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf50 = extern_kernels.convolution(buf49, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg154_1
        buf51 = buf45; del buf45  # reuse
        # Source Nodes: [shortcut_8, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf51, buf50, arg245_1, arg246_1, arg48_1, arg49_1, 6272, 32, grid=grid(6272, 32), stream=stream0)
        del arg245_1
        del arg246_1
        del arg48_1
        del arg49_1
        del buf50
        # Source Nodes: [shortcut_8, x_134, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg155_1
        del buf51
        buf53 = buf49; del buf49  # reuse
        # Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf52, arg247_1, arg248_1, arg50_1, arg51_1, buf53, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del arg247_1
        del arg248_1
        del arg50_1
        del arg51_1
        del buf52
        # Source Nodes: [x_140, x_143, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf54 = extern_kernels.convolution(buf53, arg156_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf54, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg156_1
        del buf53
        buf55 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf54, arg249_1, arg250_1, arg52_1, arg53_1, buf55, 1536, 196, grid=grid(1536, 196), stream=stream0)
        del arg249_1
        del arg250_1
        del arg52_1
        del arg53_1
        del buf54
        # Source Nodes: [x_145, x_148, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg157_1
        buf57 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf56, arg251_1, arg252_1, arg54_1, arg55_1, buf57, 512, 196, grid=grid(512, 196), stream=stream0)
        del arg251_1
        del arg252_1
        del arg54_1
        del arg55_1
        del buf56
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg158_1
        buf59 = buf55; del buf55  # reuse
        # Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf58, arg253_1, arg254_1, arg56_1, arg57_1, buf59, 1536, 196, grid=grid(1536, 196), stream=stream0)
        del arg253_1
        del arg254_1
        del arg56_1
        del arg57_1
        del buf58
        # Source Nodes: [x_156, x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg159_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf60, (8, 192, 14, 14), (37632, 196, 14, 1))
        del arg159_1
        buf61 = buf59; del buf59  # reuse
        # Source Nodes: [x_161, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf60, arg255_1, arg256_1, arg58_1, arg59_1, buf61, 1536, 196, grid=grid(1536, 196), stream=stream0)
        del arg255_1
        del arg256_1
        del arg58_1
        del arg59_1
        del buf60
        # Source Nodes: [x_161, x_164, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf62 = extern_kernels.convolution(buf61, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg160_1
        del buf61
        buf63 = buf57; del buf57  # reuse
        # Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf63, buf62, arg257_1, arg258_1, arg60_1, arg61_1, 1568, 64, grid=grid(1568, 64), stream=stream0)
        del arg257_1
        del arg258_1
        del arg60_1
        del arg61_1
        del buf62
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg161_1
        buf65 = reinterpret_tensor(buf37, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf37  # reuse
        # Source Nodes: [x_173, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf64, arg259_1, arg260_1, arg62_1, arg63_1, buf65, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg259_1
        del arg260_1
        del arg62_1
        del arg63_1
        del buf64
        # Source Nodes: [x_173, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf66 = extern_kernels.convolution(buf65, arg162_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf66, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg162_1
        buf67 = buf65; del buf65  # reuse
        # Source Nodes: [x_178, x_181], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf66, arg261_1, arg262_1, arg64_1, arg65_1, buf67, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg261_1
        del arg262_1
        del arg64_1
        del arg65_1
        del buf66
        # Source Nodes: [x_178, x_181, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf68 = extern_kernels.convolution(buf67, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg163_1
        buf69 = buf63; del buf63  # reuse
        # Source Nodes: [shortcut_11, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf69, buf68, arg263_1, arg264_1, arg66_1, arg67_1, 1568, 64, grid=grid(1568, 64), stream=stream0)
        del arg263_1
        del arg264_1
        del arg66_1
        del arg67_1
        del buf68
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg164_1
        buf71 = buf67; del buf67  # reuse
        # Source Nodes: [x_190, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf70, arg265_1, arg266_1, arg68_1, arg69_1, buf71, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg265_1
        del arg266_1
        del arg68_1
        del arg69_1
        del buf70
        # Source Nodes: [x_190, x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf72 = extern_kernels.convolution(buf71, arg165_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf72, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg165_1
        buf73 = buf71; del buf71  # reuse
        # Source Nodes: [x_195, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf72, arg267_1, arg268_1, arg70_1, arg71_1, buf73, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg267_1
        del arg268_1
        del arg70_1
        del arg71_1
        del buf72
        # Source Nodes: [x_195, x_198, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg166_1
        buf75 = buf69; del buf69  # reuse
        # Source Nodes: [shortcut_12, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf75, buf74, arg269_1, arg270_1, arg72_1, arg73_1, 1568, 64, grid=grid(1568, 64), stream=stream0)
        del arg269_1
        del arg270_1
        del arg72_1
        del arg73_1
        del buf74
        # Source Nodes: [shortcut_12, x_201, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg167_1
        del buf75
        buf77 = buf73; del buf73  # reuse
        # Source Nodes: [x_207, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf76, arg271_1, arg272_1, arg74_1, arg75_1, buf77, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg271_1
        del arg272_1
        del arg74_1
        del arg75_1
        del buf76
        # Source Nodes: [x_207, x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf77, arg168_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf78, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg168_1
        buf79 = buf77; del buf77  # reuse
        # Source Nodes: [x_212, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf78, arg273_1, arg274_1, arg76_1, arg77_1, buf79, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg273_1
        del arg274_1
        del arg76_1
        del arg77_1
        del buf78
        # Source Nodes: [x_212, x_215, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf80 = extern_kernels.convolution(buf79, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg169_1
        del buf79
        buf81 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf80, arg275_1, arg276_1, arg78_1, arg79_1, buf81, 896, 196, grid=grid(896, 196), stream=stream0)
        del arg275_1
        del arg276_1
        del arg78_1
        del arg79_1
        del buf80
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg170_1
        buf83 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf82, arg277_1, arg278_1, arg80_1, arg81_1, buf83, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg277_1
        del arg278_1
        del arg80_1
        del arg81_1
        del buf82
        # Source Nodes: [x_223, x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf84 = extern_kernels.convolution(buf83, arg171_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf84, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg171_1
        buf85 = buf83; del buf83  # reuse
        # Source Nodes: [x_228, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf84, arg279_1, arg280_1, arg82_1, arg83_1, buf85, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg279_1
        del arg280_1
        del arg82_1
        del arg83_1
        del buf84
        # Source Nodes: [x_228, x_231, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg172_1
        buf87 = buf81; del buf81  # reuse
        # Source Nodes: [shortcut_14, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf87, buf86, arg281_1, arg282_1, arg84_1, arg85_1, 1568, 112, grid=grid(1568, 112), stream=stream0)
        del arg281_1
        del arg282_1
        del arg84_1
        del arg85_1
        del buf86
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg173_1
        buf89 = buf85; del buf85  # reuse
        # Source Nodes: [x_240, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf88, arg283_1, arg284_1, arg86_1, arg87_1, buf89, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg283_1
        del arg284_1
        del arg86_1
        del arg87_1
        del buf88
        # Source Nodes: [x_240, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf90 = extern_kernels.convolution(buf89, arg174_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf90, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg174_1
        buf91 = buf89; del buf89  # reuse
        # Source Nodes: [x_245, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf90, arg285_1, arg286_1, arg88_1, arg89_1, buf91, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg285_1
        del arg286_1
        del arg88_1
        del arg89_1
        del buf90
        # Source Nodes: [x_245, x_248, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf92 = extern_kernels.convolution(buf91, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg175_1
        buf93 = buf87; del buf87  # reuse
        # Source Nodes: [shortcut_15, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf93, buf92, arg287_1, arg288_1, arg90_1, arg91_1, 1568, 112, grid=grid(1568, 112), stream=stream0)
        del arg287_1
        del arg288_1
        del arg90_1
        del arg91_1
        del buf92
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 336, 14, 14), (65856, 196, 14, 1))
        del arg176_1
        buf95 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf94, arg289_1, arg290_1, arg92_1, arg93_1, buf95, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg289_1
        del arg290_1
        del arg92_1
        del arg93_1
        del buf94
        # Source Nodes: [x_257, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf96 = extern_kernels.convolution(buf95, arg177_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf96, (8, 336, 14, 14), (65856, 196, 14, 1))
        del arg177_1
        buf97 = buf95; del buf95  # reuse
        # Source Nodes: [x_262, x_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf96, arg291_1, arg292_1, arg94_1, arg95_1, buf97, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg291_1
        del arg292_1
        del arg94_1
        del arg95_1
        del buf96
        # Source Nodes: [x_262, x_265, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf98 = extern_kernels.convolution(buf97, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg178_1
        del buf97
        buf99 = buf93; del buf93  # reuse
        # Source Nodes: [shortcut_16, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf99, buf98, arg293_1, arg294_1, arg96_1, arg97_1, 1568, 112, grid=grid(1568, 112), stream=stream0)
        del arg293_1
        del arg294_1
        del arg96_1
        del arg97_1
        del buf98
        # Source Nodes: [shortcut_16, x_268, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg179_1
        del buf99
        buf101 = buf91; del buf91  # reuse
        # Source Nodes: [x_274, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf100, arg295_1, arg296_1, arg98_1, arg99_1, buf101, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg295_1
        del arg296_1
        del arg98_1
        del arg99_1
        del buf100
        # Source Nodes: [x_274, x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf102 = extern_kernels.convolution(buf101, arg180_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf102, (8, 672, 7, 7), (32928, 49, 7, 1))
        del arg180_1
        del buf101
        buf103 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf102, arg297_1, arg298_1, arg100_1, arg101_1, buf103, 5376, 49, grid=grid(5376, 49), stream=stream0)
        del arg100_1
        del arg101_1
        del arg297_1
        del arg298_1
        del buf102
        # Source Nodes: [x_279, x_282, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf104 = extern_kernels.convolution(buf103, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 184, 7, 7), (9016, 49, 7, 1))
        del arg181_1
        del buf103
        buf105 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf104, arg299_1, arg300_1, arg102_1, arg103_1, buf105, 1472, 49, grid=grid(1472, 49), stream=stream0)
        del arg102_1
        del arg103_1
        del arg299_1
        del arg300_1
        del buf104
        # Source Nodes: [x_289], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg182_1
        buf107 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf106, arg301_1, arg302_1, arg104_1, arg105_1, buf107, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg104_1
        del arg105_1
        del arg301_1
        del arg302_1
        del buf106
        # Source Nodes: [x_290, x_293, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf108 = extern_kernels.convolution(buf107, arg183_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf108, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg183_1
        buf109 = buf107; del buf107  # reuse
        # Source Nodes: [x_295, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf108, arg303_1, arg304_1, arg106_1, arg107_1, buf109, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg106_1
        del arg107_1
        del arg303_1
        del arg304_1
        del buf108
        # Source Nodes: [x_295, x_298, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf110 = extern_kernels.convolution(buf109, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 184, 7, 7), (9016, 49, 7, 1))
        del arg184_1
        buf111 = buf105; del buf105  # reuse
        # Source Nodes: [shortcut_18, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf111, buf110, arg305_1, arg306_1, arg108_1, arg109_1, 392, 184, grid=grid(392, 184), stream=stream0)
        del arg108_1
        del arg109_1
        del arg305_1
        del arg306_1
        del buf110
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg185_1
        buf113 = buf109; del buf109  # reuse
        # Source Nodes: [x_307, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf112, arg307_1, arg308_1, arg110_1, arg111_1, buf113, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg110_1
        del arg111_1
        del arg307_1
        del arg308_1
        del buf112
        # Source Nodes: [x_307, x_310, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf114 = extern_kernels.convolution(buf113, arg186_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf114, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg186_1
        buf115 = buf113; del buf113  # reuse
        # Source Nodes: [x_312, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf114, arg309_1, arg310_1, arg112_1, arg113_1, buf115, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg112_1
        del arg113_1
        del arg309_1
        del arg310_1
        del buf114
        # Source Nodes: [x_312, x_315, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf116 = extern_kernels.convolution(buf115, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 184, 7, 7), (9016, 49, 7, 1))
        del arg187_1
        buf117 = buf111; del buf111  # reuse
        # Source Nodes: [shortcut_19, x_318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf117, buf116, arg311_1, arg312_1, arg114_1, arg115_1, 392, 184, grid=grid(392, 184), stream=stream0)
        del arg114_1
        del arg115_1
        del arg311_1
        del arg312_1
        del buf116
        # Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg188_1
        buf119 = buf115; del buf115  # reuse
        # Source Nodes: [x_324, x_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf118, arg313_1, arg314_1, arg116_1, arg117_1, buf119, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg116_1
        del arg117_1
        del arg313_1
        del arg314_1
        del buf118
        # Source Nodes: [x_324, x_327, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf120 = extern_kernels.convolution(buf119, arg189_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf120, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg189_1
        buf121 = buf119; del buf119  # reuse
        # Source Nodes: [x_329, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf120, arg315_1, arg316_1, arg118_1, arg119_1, buf121, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg118_1
        del arg119_1
        del arg315_1
        del arg316_1
        del buf120
        # Source Nodes: [x_329, x_332, x_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf122 = extern_kernels.convolution(buf121, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 184, 7, 7), (9016, 49, 7, 1))
        del arg190_1
        buf123 = buf117; del buf117  # reuse
        # Source Nodes: [shortcut_20, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf123, buf122, arg317_1, arg318_1, arg120_1, arg121_1, 392, 184, grid=grid(392, 184), stream=stream0)
        del arg120_1
        del arg121_1
        del arg317_1
        del arg318_1
        del buf122
        # Source Nodes: [shortcut_20, x_335, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg191_1
        del buf123
        buf125 = buf121; del buf121  # reuse
        # Source Nodes: [x_341, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf124, arg319_1, arg320_1, arg122_1, arg123_1, buf125, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg122_1
        del arg123_1
        del arg319_1
        del arg320_1
        del buf124
        # Source Nodes: [x_341, x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg192_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf126, (8, 1104, 7, 7), (54096, 49, 7, 1))
        del arg192_1
        buf127 = buf125; del buf125  # reuse
        # Source Nodes: [x_346, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf126, arg321_1, arg322_1, arg124_1, arg125_1, buf127, 8832, 49, grid=grid(8832, 49), stream=stream0)
        del arg124_1
        del arg125_1
        del arg321_1
        del arg322_1
        del buf126
        # Source Nodes: [x_346, x_349, x_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf128 = extern_kernels.convolution(buf127, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 352, 7, 7), (17248, 49, 7, 1))
        del arg193_1
        del buf127
        buf129 = empty_strided((8, 352, 7, 7), (17248, 1, 2464, 352), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf128, arg323_1, arg324_1, arg126_1, arg127_1, buf129, 2816, 49, grid=grid(2816, 49), stream=stream0)
        del arg126_1
        del arg127_1
        del arg323_1
        del arg324_1
        del buf128
        # Source Nodes: [x_352, x_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 1984, 7, 7), (97216, 49, 7, 1))
        del arg194_1
        del buf129
        buf131 = empty_strided((8, 1984, 1, 1), (1984, 1, 15872, 15872), device='cuda', dtype=torch.float32)
        buf132 = reinterpret_tensor(buf131, (8, 1984, 1, 1), (1984, 1, 1, 1), 0); del buf131  # reuse
        # Source Nodes: [x_358, x_362, x_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28.run(buf132, buf130, arg325_1, arg326_1, arg128_1, arg129_1, 15872, 49, grid=grid(15872), stream=stream0)
        del arg128_1
        del arg129_1
        del arg325_1
        del arg326_1
        del buf130
        buf133 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_366], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg196_1, reinterpret_tensor(buf132, (8, 1984), (1984, 1), 0), reinterpret_tensor(arg195_1, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf133)
        del arg195_1
        del arg196_1
        return (buf133, )


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
    arg8_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((96, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((96, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((32, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((112, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((336, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((112, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((184, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1104, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((352, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1984, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1000, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetc_100', benchmark_compiled_module)
