
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(78,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None)
    tmp14 = tl.load(in_ptr8 + (x2), None)
    tmp15 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr13 + (x2), None)
    tmp26 = tl.load(in_ptr14 + (x2), None)
    tmp27 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2), None)
    tmp38 = tl.load(in_ptr20 + (x2), None)
    tmp39 = tl.load(in_ptr21 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr22 + (x0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr24 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr25 + (x2), None)
    tmp50 = tl.load(in_ptr26 + (x2), None)
    tmp51 = tl.load(in_ptr27 + (x0), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr28 + (x0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr29 + (x0), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr30 + (x0), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr31 + (x2), None)
    tmp62 = tl.load(in_ptr32 + (x2), None)
    tmp63 = tl.load(in_ptr33 + (x0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr34 + (x0), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr35 + (x0), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr36 + (x0), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr37 + (x2), None)
    tmp74 = tl.load(in_ptr38 + (x2), None)
    tmp75 = tl.load(in_ptr39 + (x0), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr40 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr41 + (x0), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr42 + (x0), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr43 + (x2), None)
    tmp86 = tl.load(in_ptr44 + (x2), None)
    tmp87 = tl.load(in_ptr45 + (x0), None, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr46 + (x0), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr47 + (x0), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr48 + (x0), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr49 + (x2), None)
    tmp98 = tl.load(in_ptr50 + (x2), None)
    tmp99 = tl.load(in_ptr51 + (x0), None, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr52 + (x0), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr53 + (x0), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr54 + (x0), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr55 + (x2), None)
    tmp110 = tl.load(in_ptr56 + (x2), None)
    tmp111 = tl.load(in_ptr57 + (x0), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr58 + (x0), None, eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr59 + (x0), None, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr60 + (x0), None, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr61 + (x2), None)
    tmp122 = tl.load(in_ptr62 + (x2), None)
    tmp123 = tl.load(in_ptr63 + (x0), None, eviction_policy='evict_last')
    tmp125 = tl.load(in_ptr64 + (x0), None, eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr65 + (x0), None, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr66 + (x0), None, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 + tmp11
    tmp16 = tmp12 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp14 + tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp13 + tmp23
    tmp28 = tmp24 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp26 + tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp25 + tmp35
    tmp40 = tmp36 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp38 + tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tmp37 + tmp47
    tmp52 = tmp48 * tmp51
    tmp54 = tmp52 + tmp53
    tmp55 = tmp50 + tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tmp49 + tmp59
    tmp64 = tmp60 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp62 + tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tmp61 + tmp71
    tmp76 = tmp72 * tmp75
    tmp78 = tmp76 + tmp77
    tmp79 = tmp74 + tmp78
    tmp81 = tmp79 * tmp80
    tmp83 = tmp81 + tmp82
    tmp84 = tmp73 + tmp83
    tmp88 = tmp84 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp86 + tmp90
    tmp93 = tmp91 * tmp92
    tmp95 = tmp93 + tmp94
    tmp96 = tmp85 + tmp95
    tmp100 = tmp96 * tmp99
    tmp102 = tmp100 + tmp101
    tmp103 = tmp98 + tmp102
    tmp105 = tmp103 * tmp104
    tmp107 = tmp105 + tmp106
    tmp108 = tmp97 + tmp107
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 + tmp113
    tmp115 = tmp110 + tmp114
    tmp117 = tmp115 * tmp116
    tmp119 = tmp117 + tmp118
    tmp120 = tmp109 + tmp119
    tmp124 = tmp120 * tmp123
    tmp126 = tmp124 + tmp125
    tmp127 = tmp122 + tmp126
    tmp129 = tmp127 * tmp128
    tmp131 = tmp129 + tmp130
    tmp132 = tmp121 + tmp131
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp24, None)
    tl.store(out_ptr2 + (x2), tmp36, None)
    tl.store(out_ptr3 + (x2), tmp48, None)
    tl.store(out_ptr4 + (x2), tmp60, None)
    tl.store(out_ptr5 + (x2), tmp72, None)
    tl.store(out_ptr6 + (x2), tmp84, None)
    tl.store(out_ptr7 + (x2), tmp96, None)
    tl.store(out_ptr8 + (x2), tmp108, None)
    tl.store(out_ptr9 + (x2), tmp120, None)
    tl.store(out_ptr10 + (x2), tmp132, None)
