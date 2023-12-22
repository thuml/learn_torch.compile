
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 200
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (200*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (39200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (200*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.0006377551020408163
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (200*y3)), tmp29, xmask & ymask)
