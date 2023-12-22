
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 1568)
    x3 = xindex % 1568
    x1 = (xindex // 49) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (42336 + x3 + (50176*x2)), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (864 + x1 + (1024*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (42336 + x3 + (48608*x2)), xmask)
    tmp15 = tl.load(in_ptr5 + (42336 + x3 + (48608*x2)), xmask)
    tmp17 = tl.load(in_ptr6 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (42336 + x3 + (47040*x2)), xmask)
    tmp26 = tl.load(in_ptr9 + (42336 + x3 + (47040*x2)), xmask)
    tmp28 = tl.load(in_ptr10 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr11 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr12 + (42336 + x3 + (45472*x2)), xmask)
    tmp37 = tl.load(in_ptr13 + (42336 + x3 + (45472*x2)), xmask)
    tmp39 = tl.load(in_ptr14 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr15 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr16 + (42336 + x3 + (43904*x2)), xmask)
    tmp48 = tl.load(in_ptr17 + (42336 + x3 + (43904*x2)), xmask)
    tmp50 = tl.load(in_ptr18 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr19 + (864 + x1), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13 <= tmp4
    tmp16 = tl.where(tmp14, tmp4, tmp15)
    tmp18 = tmp17 + tmp7
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp12 + tmp22
    tmp25 = tmp24 <= tmp4
    tmp27 = tl.where(tmp25, tmp4, tmp26)
    tmp29 = tmp28 + tmp7
    tmp30 = tl.math.rsqrt(tmp29)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp27 * tmp32
    tmp34 = tmp23 + tmp33
    tmp36 = tmp35 <= tmp4
    tmp38 = tl.where(tmp36, tmp4, tmp37)
    tmp40 = tmp39 + tmp7
    tmp41 = tl.math.rsqrt(tmp40)
    tmp43 = tmp41 * tmp42
    tmp44 = tmp38 * tmp43
    tmp45 = tmp34 + tmp44
    tmp47 = tmp46 <= tmp4
    tmp49 = tl.where(tmp47, tmp4, tmp48)
    tmp51 = tmp50 + tmp7
    tmp52 = tl.math.rsqrt(tmp51)
    tmp54 = tmp52 * tmp53
    tmp55 = tmp49 * tmp54
    tmp56 = tmp45 + tmp55
    tl.store(in_out_ptr0 + (x4), tmp56, xmask)
