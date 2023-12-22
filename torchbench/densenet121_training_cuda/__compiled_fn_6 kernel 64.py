
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(77,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6272)
    x3 = xindex % 6272
    x1 = (xindex // 196) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (81536 + x3 + (200704*x2)), xmask)
    tmp3 = tl.load(in_ptr1 + (81536 + x3 + (200704*x2)), xmask)
    tmp5 = tl.load(in_ptr2 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (81536 + x3 + (194432*x2)), xmask)
    tmp14 = tl.load(in_ptr5 + (81536 + x3 + (194432*x2)), xmask)
    tmp16 = tl.load(in_ptr6 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (81536 + x3 + (188160*x2)), xmask)
    tmp25 = tl.load(in_ptr9 + (81536 + x3 + (188160*x2)), xmask)
    tmp27 = tl.load(in_ptr10 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr12 + (81536 + x3 + (181888*x2)), xmask)
    tmp36 = tl.load(in_ptr13 + (81536 + x3 + (181888*x2)), xmask)
    tmp38 = tl.load(in_ptr14 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr15 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr16 + (81536 + x3 + (175616*x2)), xmask)
    tmp47 = tl.load(in_ptr17 + (81536 + x3 + (175616*x2)), xmask)
    tmp49 = tl.load(in_ptr18 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr19 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr20 + (81536 + x3 + (169344*x2)), xmask)
    tmp58 = tl.load(in_ptr21 + (81536 + x3 + (169344*x2)), xmask)
    tmp60 = tl.load(in_ptr22 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr23 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr24 + (81536 + x3 + (163072*x2)), xmask)
    tmp69 = tl.load(in_ptr25 + (81536 + x3 + (163072*x2)), xmask)
    tmp71 = tl.load(in_ptr26 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr27 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr28 + (81536 + x3 + (156800*x2)), xmask)
    tmp80 = tl.load(in_ptr29 + (81536 + x3 + (156800*x2)), xmask)
    tmp82 = tl.load(in_ptr30 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr31 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr32 + (81536 + x3 + (150528*x2)), xmask)
    tmp91 = tl.load(in_ptr33 + (81536 + x3 + (150528*x2)), xmask)
    tmp93 = tl.load(in_ptr34 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr35 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr36 + (81536 + x3 + (144256*x2)), xmask)
    tmp102 = tl.load(in_ptr37 + (81536 + x3 + (144256*x2)), xmask)
    tmp104 = tl.load(in_ptr38 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr39 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr40 + (81536 + x3 + (137984*x2)), xmask)
    tmp113 = tl.load(in_ptr41 + (81536 + x3 + (137984*x2)), xmask)
    tmp115 = tl.load(in_ptr42 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr43 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr44 + (81536 + x3 + (131712*x2)), xmask)
    tmp124 = tl.load(in_ptr45 + (81536 + x3 + (131712*x2)), xmask)
    tmp126 = tl.load(in_ptr46 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr47 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr48 + (81536 + x3 + (125440*x2)), xmask)
    tmp135 = tl.load(in_ptr49 + (81536 + x3 + (125440*x2)), xmask)
    tmp137 = tl.load(in_ptr50 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp140 = tl.load(in_ptr51 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr52 + (81536 + x3 + (119168*x2)), xmask)
    tmp146 = tl.load(in_ptr53 + (81536 + x3 + (119168*x2)), xmask)
    tmp148 = tl.load(in_ptr54 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp151 = tl.load(in_ptr55 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp155 = tl.load(in_ptr56 + (81536 + x3 + (112896*x2)), xmask)
    tmp157 = tl.load(in_ptr57 + (81536 + x3 + (112896*x2)), xmask)
    tmp159 = tl.load(in_ptr58 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr59 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp166 = tl.load(in_ptr60 + (81536 + x3 + (106624*x2)), xmask)
    tmp168 = tl.load(in_ptr61 + (81536 + x3 + (106624*x2)), xmask)
    tmp170 = tl.load(in_ptr62 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr63 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp177 = tl.load(in_ptr64 + (81536 + x3 + (100352*x2)), xmask)
    tmp179 = tl.load(in_ptr65 + (81536 + x3 + (100352*x2)), xmask)
    tmp181 = tl.load(in_ptr66 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp184 = tl.load(in_ptr67 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp188 = tl.load(in_ptr68 + (81536 + x3 + (94080*x2)), xmask)
    tmp190 = tl.load(in_ptr69 + (81536 + x3 + (94080*x2)), xmask)
    tmp192 = tl.load(in_ptr70 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp195 = tl.load(in_ptr71 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr72 + (81536 + x3 + (87808*x2)), xmask)
    tmp201 = tl.load(in_ptr73 + (81536 + x3 + (87808*x2)), xmask)
    tmp203 = tl.load(in_ptr74 + (416 + x1), xmask, eviction_policy='evict_last')
    tmp206 = tl.load(in_ptr75 + (416 + x1), xmask, eviction_policy='evict_last')
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
    tmp101 = tmp100 <= tmp1
    tmp103 = tl.where(tmp101, tmp1, tmp102)
    tmp105 = tmp104 + tmp6
    tmp106 = tl.math.rsqrt(tmp105)
    tmp108 = tmp106 * tmp107
    tmp109 = tmp103 * tmp108
    tmp110 = tmp99 + tmp109
    tmp112 = tmp111 <= tmp1
    tmp114 = tl.where(tmp112, tmp1, tmp113)
    tmp116 = tmp115 + tmp6
    tmp117 = tl.math.rsqrt(tmp116)
    tmp119 = tmp117 * tmp118
    tmp120 = tmp114 * tmp119
    tmp121 = tmp110 + tmp120
    tmp123 = tmp122 <= tmp1
    tmp125 = tl.where(tmp123, tmp1, tmp124)
    tmp127 = tmp126 + tmp6
    tmp128 = tl.math.rsqrt(tmp127)
    tmp130 = tmp128 * tmp129
    tmp131 = tmp125 * tmp130
    tmp132 = tmp121 + tmp131
    tmp134 = tmp133 <= tmp1
    tmp136 = tl.where(tmp134, tmp1, tmp135)
    tmp138 = tmp137 + tmp6
    tmp139 = tl.math.rsqrt(tmp138)
    tmp141 = tmp139 * tmp140
    tmp142 = tmp136 * tmp141
    tmp143 = tmp132 + tmp142
    tmp145 = tmp144 <= tmp1
    tmp147 = tl.where(tmp145, tmp1, tmp146)
    tmp149 = tmp148 + tmp6
    tmp150 = tl.math.rsqrt(tmp149)
    tmp152 = tmp150 * tmp151
    tmp153 = tmp147 * tmp152
    tmp154 = tmp143 + tmp153
    tmp156 = tmp155 <= tmp1
    tmp158 = tl.where(tmp156, tmp1, tmp157)
    tmp160 = tmp159 + tmp6
    tmp161 = tl.math.rsqrt(tmp160)
    tmp163 = tmp161 * tmp162
    tmp164 = tmp158 * tmp163
    tmp165 = tmp154 + tmp164
    tmp167 = tmp166 <= tmp1
    tmp169 = tl.where(tmp167, tmp1, tmp168)
    tmp171 = tmp170 + tmp6
    tmp172 = tl.math.rsqrt(tmp171)
    tmp174 = tmp172 * tmp173
    tmp175 = tmp169 * tmp174
    tmp176 = tmp165 + tmp175
    tmp178 = tmp177 <= tmp1
    tmp180 = tl.where(tmp178, tmp1, tmp179)
    tmp182 = tmp181 + tmp6
    tmp183 = tl.math.rsqrt(tmp182)
    tmp185 = tmp183 * tmp184
    tmp186 = tmp180 * tmp185
    tmp187 = tmp176 + tmp186
    tmp189 = tmp188 <= tmp1
    tmp191 = tl.where(tmp189, tmp1, tmp190)
    tmp193 = tmp192 + tmp6
    tmp194 = tl.math.rsqrt(tmp193)
    tmp196 = tmp194 * tmp195
    tmp197 = tmp191 * tmp196
    tmp198 = tmp187 + tmp197
    tmp200 = tmp199 <= tmp1
    tmp202 = tl.where(tmp200, tmp1, tmp201)
    tmp204 = tmp203 + tmp6
    tmp205 = tl.math.rsqrt(tmp204)
    tmp207 = tmp205 * tmp206
    tmp208 = tmp202 * tmp207
    tmp209 = tmp198 + tmp208
    tl.store(in_out_ptr0 + (x4), tmp209, xmask)
