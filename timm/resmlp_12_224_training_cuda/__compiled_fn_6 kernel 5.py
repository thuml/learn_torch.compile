
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: 'i32', 60: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(59, 60))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr14 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr16 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr17 + (x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr24 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr26 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr27 + (x2), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr28 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr29 + (x2), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr30 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr31 + (x2), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr32 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr33 + (x2), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr34 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr35 + (x2), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr36 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr37 + (x2), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr38 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr39 + (x2), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr40 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr41 + (x2 + (384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr42 + (x2), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr43 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr44 + (x2), xmask, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr45 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr46 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp31 = tmp29 * tmp30
    tmp32 = tmp28 + tmp31
    tmp35 = tmp33 * tmp34
    tmp36 = tmp32 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp43 = tmp41 * tmp42
    tmp44 = tmp40 + tmp43
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 + tmp47
    tmp51 = tmp49 * tmp50
    tmp52 = tmp48 + tmp51
    tmp55 = tmp53 * tmp54
    tmp56 = tmp52 + tmp55
    tmp59 = tmp57 * tmp58
    tmp60 = tmp56 + tmp59
    tmp63 = tmp61 * tmp62
    tmp64 = tmp60 + tmp63
    tmp67 = tmp65 * tmp66
    tmp68 = tmp64 + tmp67
    tmp71 = tmp69 * tmp70
    tmp72 = tmp68 + tmp71
    tmp75 = tmp73 * tmp74
    tmp76 = tmp72 + tmp75
    tmp79 = tmp77 * tmp78
    tmp80 = tmp76 + tmp79
    tmp82 = 196.0
    tmp83 = tmp81 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp83 * tmp86
    tmp90 = tmp89 * tmp85
    tmp91 = tmp88 * tmp90
    tmp92 = tmp87 + tmp91
    tmp95 = tmp94 * tmp85
    tmp96 = tmp93 * tmp95
    tmp97 = tmp92 + tmp96
    tmp98 = tmp97 * tmp77
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp8, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (384*y3)), tmp16, xmask & ymask)
    tl.store(out_ptr2 + (x2 + (384*y3)), tmp24, xmask & ymask)
    tl.store(out_ptr3 + (x2 + (384*y3)), tmp32, xmask & ymask)
    tl.store(out_ptr4 + (x2 + (384*y3)), tmp40, xmask & ymask)
    tl.store(out_ptr5 + (x2 + (384*y3)), tmp48, xmask & ymask)
    tl.store(out_ptr6 + (x2 + (384*y3)), tmp56, xmask & ymask)
    tl.store(out_ptr7 + (x2 + (384*y3)), tmp64, xmask & ymask)
    tl.store(out_ptr8 + (x2 + (384*y3)), tmp72, xmask & ymask)
    tl.store(out_ptr9 + (x2 + (384*y3)), tmp80, xmask & ymask)
    tl.store(out_ptr10 + (x2 + (384*y3)), tmp97, xmask & ymask)
    tl.store(out_ptr11 + (x2 + (384*y3)), tmp98, xmask & ymask)
