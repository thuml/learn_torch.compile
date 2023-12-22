
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: 'i32', 104: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(103, 104))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr14 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr24 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr26 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr27 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr28 + (x2), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr29 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr30 + (x2), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr31 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr32 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr33 + (x2), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr34 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr35 + (x2), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr36 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr37 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr38 + (x2), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr39 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr40 + (x2), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr41 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr42 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr43 + (x2), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr44 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr45 + (x2), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr46 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr47 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr48 + (x2), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr49 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr50 + (x2), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr51 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr52 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr53 + (x2), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr54 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr55 + (x2), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr56 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr57 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr58 + (x2), xmask, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr59 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr60 + (x2), xmask, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr61 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr62 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr63 + (x2), xmask, eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr64 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr65 + (x2), xmask, eviction_policy='evict_last')
    tmp131 = tl.load(in_ptr66 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp132 = tl.load(in_ptr67 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr68 + (x2), xmask, eviction_policy='evict_last')
    tmp137 = tl.load(in_ptr69 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr70 + (x2), xmask, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr71 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp142 = tl.load(in_ptr72 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr73 + (x2), xmask, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr74 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp148 = tl.load(in_ptr75 + (x2), xmask, eviction_policy='evict_last')
    tmp151 = tl.load(in_ptr76 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr77 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr78 + (x2), xmask, eviction_policy='evict_last')
    tmp157 = tl.load(in_ptr79 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr80 + (x2), xmask, eviction_policy='evict_last')
    tmp161 = tl.load(in_ptr81 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr82 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp164 = tl.load(in_ptr83 + (x2), xmask, eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr84 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr85 + (x2), xmask, eviction_policy='evict_last')
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
    tmp53 = tmp51 - tmp52
    tmp55 = tmp53 * tmp54
    tmp56 = tmp50 + tmp55
    tmp59 = tmp57 * tmp58
    tmp60 = tmp56 + tmp59
    tmp63 = tmp61 - tmp62
    tmp65 = tmp63 * tmp64
    tmp66 = tmp60 + tmp65
    tmp69 = tmp67 * tmp68
    tmp70 = tmp66 + tmp69
    tmp73 = tmp71 - tmp72
    tmp75 = tmp73 * tmp74
    tmp76 = tmp70 + tmp75
    tmp79 = tmp77 * tmp78
    tmp80 = tmp76 + tmp79
    tmp83 = tmp81 - tmp82
    tmp85 = tmp83 * tmp84
    tmp86 = tmp80 + tmp85
    tmp89 = tmp87 * tmp88
    tmp90 = tmp86 + tmp89
    tmp93 = tmp91 - tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tmp90 + tmp95
    tmp99 = tmp97 * tmp98
    tmp100 = tmp96 + tmp99
    tmp103 = tmp101 - tmp102
    tmp105 = tmp103 * tmp104
    tmp106 = tmp100 + tmp105
    tmp109 = tmp107 * tmp108
    tmp110 = tmp106 + tmp109
    tmp113 = tmp111 - tmp112
    tmp115 = tmp113 * tmp114
    tmp116 = tmp110 + tmp115
    tmp119 = tmp117 * tmp118
    tmp120 = tmp116 + tmp119
    tmp123 = tmp121 - tmp122
    tmp125 = tmp123 * tmp124
    tmp126 = tmp120 + tmp125
    tmp129 = tmp127 * tmp128
    tmp130 = tmp126 + tmp129
    tmp133 = tmp131 - tmp132
    tmp135 = tmp133 * tmp134
    tmp136 = tmp130 + tmp135
    tmp139 = tmp137 * tmp138
    tmp140 = tmp136 + tmp139
    tmp143 = tmp141 - tmp142
    tmp145 = tmp143 * tmp144
    tmp146 = tmp140 + tmp145
    tmp149 = tmp147 * tmp148
    tmp150 = tmp146 + tmp149
    tmp153 = tmp151 - tmp152
    tmp155 = tmp153 * tmp154
    tmp156 = tmp150 + tmp155
    tmp159 = tmp157 * tmp158
    tmp160 = tmp156 + tmp159
    tmp163 = tmp161 - tmp162
    tmp165 = tmp163 * tmp164
    tmp166 = tmp160 + tmp165
    tmp169 = tmp167 * tmp168
    tmp170 = tmp166 + tmp169
    tl.store(out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (196*x2) + (75264*y1)), tmp20, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (196*x2) + (75264*y1)), tmp30, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (196*x2) + (75264*y1)), tmp40, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (196*x2) + (75264*y1)), tmp50, xmask & ymask)
    tl.store(out_ptr5 + (y0 + (196*x2) + (75264*y1)), tmp60, xmask & ymask)
    tl.store(out_ptr6 + (y0 + (196*x2) + (75264*y1)), tmp70, xmask & ymask)
    tl.store(out_ptr7 + (y0 + (196*x2) + (75264*y1)), tmp80, xmask & ymask)
    tl.store(out_ptr8 + (y0 + (196*x2) + (75264*y1)), tmp90, xmask & ymask)
    tl.store(out_ptr9 + (y0 + (196*x2) + (75264*y1)), tmp100, xmask & ymask)
    tl.store(out_ptr10 + (y0 + (196*x2) + (75264*y1)), tmp110, xmask & ymask)
    tl.store(out_ptr11 + (y0 + (196*x2) + (75264*y1)), tmp120, xmask & ymask)
    tl.store(out_ptr12 + (y0 + (196*x2) + (75264*y1)), tmp130, xmask & ymask)
    tl.store(out_ptr13 + (y0 + (196*x2) + (75264*y1)), tmp140, xmask & ymask)
    tl.store(out_ptr14 + (y0 + (196*x2) + (75264*y1)), tmp150, xmask & ymask)
    tl.store(out_ptr15 + (y0 + (196*x2) + (75264*y1)), tmp160, xmask & ymask)
    tl.store(out_ptr16 + (y0 + (196*x2) + (75264*y1)), tmp170, xmask & ymask)
