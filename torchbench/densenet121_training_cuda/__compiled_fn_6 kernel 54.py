
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(37,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6272)
    x3 = xindex % 6272
    x1 = (xindex // 196) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (144256 + x3 + (200704*x2)), xmask)
    tmp3 = tl.load(in_ptr1 + (144256 + x3 + (200704*x2)), xmask)
    tmp5 = tl.load(in_ptr2 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (144256 + x3 + (194432*x2)), xmask)
    tmp14 = tl.load(in_ptr5 + (144256 + x3 + (194432*x2)), xmask)
    tmp16 = tl.load(in_ptr6 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (144256 + x3 + (188160*x2)), xmask)
    tmp25 = tl.load(in_ptr9 + (144256 + x3 + (188160*x2)), xmask)
    tmp27 = tl.load(in_ptr10 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr12 + (144256 + x3 + (181888*x2)), xmask)
    tmp36 = tl.load(in_ptr13 + (144256 + x3 + (181888*x2)), xmask)
    tmp38 = tl.load(in_ptr14 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr15 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr16 + (144256 + x3 + (175616*x2)), xmask)
    tmp47 = tl.load(in_ptr17 + (144256 + x3 + (175616*x2)), xmask)
    tmp49 = tl.load(in_ptr18 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr19 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr20 + (144256 + x3 + (169344*x2)), xmask)
    tmp58 = tl.load(in_ptr21 + (144256 + x3 + (169344*x2)), xmask)
    tmp60 = tl.load(in_ptr22 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr23 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr24 + (144256 + x3 + (163072*x2)), xmask)
    tmp69 = tl.load(in_ptr25 + (144256 + x3 + (163072*x2)), xmask)
    tmp71 = tl.load(in_ptr26 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr27 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr28 + (144256 + x3 + (156800*x2)), xmask)
    tmp80 = tl.load(in_ptr29 + (144256 + x3 + (156800*x2)), xmask)
    tmp82 = tl.load(in_ptr30 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr31 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr32 + (144256 + x3 + (150528*x2)), xmask)
    tmp91 = tl.load(in_ptr33 + (144256 + x3 + (150528*x2)), xmask)
    tmp93 = tl.load(in_ptr34 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr35 + (736 + x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tmp13 = tmp12 <= tmp1
    tmp15 = tl.where(tmp13, tmp1, tmp14)
    tmp17 = tmp16 + tmp6
    tmp18 = tl.math.rsqrt(tmp17)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp11 + tmp21
    tmp24 = tmp23 <= tmp1
    tmp26 = tl.where(tmp24, tmp1, tmp25)
    tmp28 = tmp27 + tmp6
    tmp29 = tl.math.rsqrt(tmp28)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp22 + tmp32
    tmp35 = tmp34 <= tmp1
    tmp37 = tl.where(tmp35, tmp1, tmp36)
    tmp39 = tmp38 + tmp6
    tmp40 = tl.math.rsqrt(tmp39)
    tmp42 = tmp40 * tmp41
    tmp43 = tmp37 * tmp42
    tmp44 = tmp33 + tmp43
    tmp46 = tmp45 <= tmp1
    tmp48 = tl.where(tmp46, tmp1, tmp47)
    tmp50 = tmp49 + tmp6
    tmp51 = tl.math.rsqrt(tmp50)
    tmp53 = tmp51 * tmp52
    tmp54 = tmp48 * tmp53
    tmp55 = tmp44 + tmp54
    tmp57 = tmp56 <= tmp1
    tmp59 = tl.where(tmp57, tmp1, tmp58)
    tmp61 = tmp60 + tmp6
    tmp62 = tl.math.rsqrt(tmp61)
    tmp64 = tmp62 * tmp63
    tmp65 = tmp59 * tmp64
    tmp66 = tmp55 + tmp65
    tmp68 = tmp67 <= tmp1
    tmp70 = tl.where(tmp68, tmp1, tmp69)
    tmp72 = tmp71 + tmp6
    tmp73 = tl.math.rsqrt(tmp72)
    tmp75 = tmp73 * tmp74
    tmp76 = tmp70 * tmp75
    tmp77 = tmp66 + tmp76
    tmp79 = tmp78 <= tmp1
    tmp81 = tl.where(tmp79, tmp1, tmp80)
    tmp83 = tmp82 + tmp6
    tmp84 = tl.math.rsqrt(tmp83)
    tmp86 = tmp84 * tmp85
    tmp87 = tmp81 * tmp86
    tmp88 = tmp77 + tmp87
    tmp90 = tmp89 <= tmp1
    tmp92 = tl.where(tmp90, tmp1, tmp91)
    tmp94 = tmp93 + tmp6
    tmp95 = tl.math.rsqrt(tmp94)
    tmp97 = tmp95 * tmp96
    tmp98 = tmp92 * tmp97
    tmp99 = tmp88 + tmp98
    tl.store(in_out_ptr0 + (x4), tmp99, xmask)
