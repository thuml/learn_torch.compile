
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(37,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 1568)
    x3 = xindex % 1568
    x1 = (xindex // 49) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (36064 + x3 + (50176*x2)), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (736 + x1 + (1024*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (36064 + x3 + (48608*x2)), xmask)
    tmp15 = tl.load(in_ptr5 + (36064 + x3 + (48608*x2)), xmask)
    tmp17 = tl.load(in_ptr6 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (36064 + x3 + (47040*x2)), xmask)
    tmp26 = tl.load(in_ptr9 + (36064 + x3 + (47040*x2)), xmask)
    tmp28 = tl.load(in_ptr10 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr11 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr12 + (36064 + x3 + (45472*x2)), xmask)
    tmp37 = tl.load(in_ptr13 + (36064 + x3 + (45472*x2)), xmask)
    tmp39 = tl.load(in_ptr14 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr15 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr16 + (36064 + x3 + (43904*x2)), xmask)
    tmp48 = tl.load(in_ptr17 + (36064 + x3 + (43904*x2)), xmask)
    tmp50 = tl.load(in_ptr18 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr19 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr20 + (36064 + x3 + (42336*x2)), xmask)
    tmp59 = tl.load(in_ptr21 + (36064 + x3 + (42336*x2)), xmask)
    tmp61 = tl.load(in_ptr22 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr23 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr24 + (36064 + x3 + (40768*x2)), xmask)
    tmp70 = tl.load(in_ptr25 + (36064 + x3 + (40768*x2)), xmask)
    tmp72 = tl.load(in_ptr26 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr27 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr28 + (36064 + x3 + (39200*x2)), xmask)
    tmp81 = tl.load(in_ptr29 + (36064 + x3 + (39200*x2)), xmask)
    tmp83 = tl.load(in_ptr30 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr31 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr32 + (36064 + x3 + (37632*x2)), xmask)
    tmp92 = tl.load(in_ptr33 + (36064 + x3 + (37632*x2)), xmask)
    tmp94 = tl.load(in_ptr34 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr35 + (736 + x1), xmask, eviction_policy='evict_last')
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
    tmp58 = tmp57 <= tmp4
    tmp60 = tl.where(tmp58, tmp4, tmp59)
    tmp62 = tmp61 + tmp7
    tmp63 = tl.math.rsqrt(tmp62)
    tmp65 = tmp63 * tmp64
    tmp66 = tmp60 * tmp65
    tmp67 = tmp56 + tmp66
    tmp69 = tmp68 <= tmp4
    tmp71 = tl.where(tmp69, tmp4, tmp70)
    tmp73 = tmp72 + tmp7
    tmp74 = tl.math.rsqrt(tmp73)
    tmp76 = tmp74 * tmp75
    tmp77 = tmp71 * tmp76
    tmp78 = tmp67 + tmp77
    tmp80 = tmp79 <= tmp4
    tmp82 = tl.where(tmp80, tmp4, tmp81)
    tmp84 = tmp83 + tmp7
    tmp85 = tl.math.rsqrt(tmp84)
    tmp87 = tmp85 * tmp86
    tmp88 = tmp82 * tmp87
    tmp89 = tmp78 + tmp88
    tmp91 = tmp90 <= tmp4
    tmp93 = tl.where(tmp91, tmp4, tmp92)
    tmp95 = tmp94 + tmp7
    tmp96 = tl.math.rsqrt(tmp95)
    tmp98 = tmp96 * tmp97
    tmp99 = tmp93 * tmp98
    tmp100 = tmp89 + tmp99
    tl.store(in_out_ptr0 + (x4), tmp100, xmask)
