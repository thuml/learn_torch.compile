
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (65536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (64*x2) + (65536*y1)), tmp13, xmask & ymask)
