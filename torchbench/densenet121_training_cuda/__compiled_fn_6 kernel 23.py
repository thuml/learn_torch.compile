
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(65,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 1568)
    x3 = xindex % 1568
    x1 = (xindex // 49) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (25088 + x3 + (50176*x2)), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (25088 + x3 + (48608*x2)), xmask)
    tmp15 = tl.load(in_ptr5 + (25088 + x3 + (48608*x2)), xmask)
    tmp17 = tl.load(in_ptr6 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (25088 + x3 + (47040*x2)), xmask)
    tmp26 = tl.load(in_ptr9 + (25088 + x3 + (47040*x2)), xmask)
    tmp28 = tl.load(in_ptr10 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr11 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr12 + (25088 + x3 + (45472*x2)), xmask)
    tmp37 = tl.load(in_ptr13 + (25088 + x3 + (45472*x2)), xmask)
    tmp39 = tl.load(in_ptr14 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr15 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr16 + (25088 + x3 + (43904*x2)), xmask)
    tmp48 = tl.load(in_ptr17 + (25088 + x3 + (43904*x2)), xmask)
    tmp50 = tl.load(in_ptr18 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr19 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr20 + (25088 + x3 + (42336*x2)), xmask)
    tmp59 = tl.load(in_ptr21 + (25088 + x3 + (42336*x2)), xmask)
    tmp61 = tl.load(in_ptr22 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr23 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr24 + (25088 + x3 + (40768*x2)), xmask)
    tmp70 = tl.load(in_ptr25 + (25088 + x3 + (40768*x2)), xmask)
    tmp72 = tl.load(in_ptr26 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr27 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr28 + (25088 + x3 + (39200*x2)), xmask)
    tmp81 = tl.load(in_ptr29 + (25088 + x3 + (39200*x2)), xmask)
    tmp83 = tl.load(in_ptr30 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr31 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr32 + (25088 + x3 + (37632*x2)), xmask)
    tmp92 = tl.load(in_ptr33 + (25088 + x3 + (37632*x2)), xmask)
    tmp94 = tl.load(in_ptr34 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr35 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr36 + (25088 + x3 + (36064*x2)), xmask)
    tmp103 = tl.load(in_ptr37 + (25088 + x3 + (36064*x2)), xmask)
    tmp105 = tl.load(in_ptr38 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr39 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr40 + (25088 + x3 + (34496*x2)), xmask)
    tmp114 = tl.load(in_ptr41 + (25088 + x3 + (34496*x2)), xmask)
    tmp116 = tl.load(in_ptr42 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr43 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr44 + (25088 + x3 + (32928*x2)), xmask)
    tmp125 = tl.load(in_ptr45 + (25088 + x3 + (32928*x2)), xmask)
    tmp127 = tl.load(in_ptr46 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr47 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr48 + (25088 + x3 + (31360*x2)), xmask)
    tmp136 = tl.load(in_ptr49 + (25088 + x3 + (31360*x2)), xmask)
    tmp138 = tl.load(in_ptr50 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr51 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr52 + (25088 + x3 + (29792*x2)), xmask)
    tmp147 = tl.load(in_ptr53 + (25088 + x3 + (29792*x2)), xmask)
    tmp149 = tl.load(in_ptr54 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr55 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr56 + (25088 + x3 + (28224*x2)), xmask)
    tmp158 = tl.load(in_ptr57 + (25088 + x3 + (28224*x2)), xmask)
    tmp160 = tl.load(in_ptr58 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp163 = tl.load(in_ptr59 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr60 + (25088 + x3 + (26656*x2)), xmask)
    tmp169 = tl.load(in_ptr61 + (25088 + x3 + (26656*x2)), xmask)
    tmp171 = tl.load(in_ptr62 + (512 + x1), xmask, eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr63 + (512 + x1), xmask, eviction_policy='evict_last')
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
    tmp102 = tmp101 <= tmp4
    tmp104 = tl.where(tmp102, tmp4, tmp103)
    tmp106 = tmp105 + tmp7
    tmp107 = tl.math.rsqrt(tmp106)
    tmp109 = tmp107 * tmp108
    tmp110 = tmp104 * tmp109
    tmp111 = tmp100 + tmp110
    tmp113 = tmp112 <= tmp4
    tmp115 = tl.where(tmp113, tmp4, tmp114)
    tmp117 = tmp116 + tmp7
    tmp118 = tl.math.rsqrt(tmp117)
    tmp120 = tmp118 * tmp119
    tmp121 = tmp115 * tmp120
    tmp122 = tmp111 + tmp121
    tmp124 = tmp123 <= tmp4
    tmp126 = tl.where(tmp124, tmp4, tmp125)
    tmp128 = tmp127 + tmp7
    tmp129 = tl.math.rsqrt(tmp128)
    tmp131 = tmp129 * tmp130
    tmp132 = tmp126 * tmp131
    tmp133 = tmp122 + tmp132
    tmp135 = tmp134 <= tmp4
    tmp137 = tl.where(tmp135, tmp4, tmp136)
    tmp139 = tmp138 + tmp7
    tmp140 = tl.math.rsqrt(tmp139)
    tmp142 = tmp140 * tmp141
    tmp143 = tmp137 * tmp142
    tmp144 = tmp133 + tmp143
    tmp146 = tmp145 <= tmp4
    tmp148 = tl.where(tmp146, tmp4, tmp147)
    tmp150 = tmp149 + tmp7
    tmp151 = tl.math.rsqrt(tmp150)
    tmp153 = tmp151 * tmp152
    tmp154 = tmp148 * tmp153
    tmp155 = tmp144 + tmp154
    tmp157 = tmp156 <= tmp4
    tmp159 = tl.where(tmp157, tmp4, tmp158)
    tmp161 = tmp160 + tmp7
    tmp162 = tl.math.rsqrt(tmp161)
    tmp164 = tmp162 * tmp163
    tmp165 = tmp159 * tmp164
    tmp166 = tmp155 + tmp165
    tmp168 = tmp167 <= tmp4
    tmp170 = tl.where(tmp168, tmp4, tmp169)
    tmp172 = tmp171 + tmp7
    tmp173 = tl.math.rsqrt(tmp172)
    tmp175 = tmp173 * tmp174
    tmp176 = tmp170 * tmp175
    tmp177 = tmp166 + tmp176
    tl.store(in_out_ptr0 + (x4), tmp177, xmask)
