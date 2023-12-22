
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x2 + (2048*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 * tmp21
    tmp24 = tmp23 + tmp17
    tmp25 = tl.math.rsqrt(tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tl.store(out_ptr0 + (x2 + (2048*y3)), tmp15, ymask)
    tl.store(out_ptr1 + (x2 + (2048*y3)), tmp22, ymask)
    tl.store(out_ptr2 + (x2 + (2048*y3)), tmp28, ymask)
