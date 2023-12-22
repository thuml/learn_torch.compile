
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_relu_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (786432*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp1 - tmp7
    tmp10 = 0.0001220703125
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp6 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tl.where(tmp3, tmp2, tmp22)
    tl.store(out_ptr0 + (y0 + (768*x2) + (786432*y1)), tmp23, xmask)
