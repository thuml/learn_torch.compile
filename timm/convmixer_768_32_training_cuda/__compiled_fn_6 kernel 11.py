
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_relu_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (y0 + (1024*x2) + (786432*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp1 - tmp5
    tmp8 = 0.0001220703125
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp4 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tl.where(tmp3, tmp2, tmp20)
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp21, xmask)
