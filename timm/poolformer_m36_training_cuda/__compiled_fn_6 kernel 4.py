
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: 'i32', 32: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(31, 32))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr14 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr24 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 - tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp20 + tmp25
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp33 = tmp31 - tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp30 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp43 = tmp41 - tmp42
    tmp45 = tmp43 * tmp44
    tmp46 = tmp40 + tmp45
    tmp49 = tmp47 * tmp48
    tmp50 = tmp46 + tmp49
    tl.store(out_ptr0 + (y0 + (49*x2) + (37632*y1)), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (49*x2) + (37632*y1)), tmp20, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (49*x2) + (37632*y1)), tmp30, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (49*x2) + (37632*y1)), tmp40, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (49*x2) + (37632*y1)), tmp50, xmask & ymask)
