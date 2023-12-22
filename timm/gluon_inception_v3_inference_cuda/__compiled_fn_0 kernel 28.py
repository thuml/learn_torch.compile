
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    x5 = (xindex // 17)
    x4 = xindex % 17
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x5
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 17, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x4
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-18) + x2 + (289*y3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = x4
    tmp16 = tmp15 >= tmp2
    tmp17 = tmp15 < tmp4
    tmp18 = tmp16 & tmp17
    tmp19 = tmp6 & tmp18
    tmp20 = tl.load(in_ptr0 + ((-17) + x2 + (289*y3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp22 + tmp14
    tmp24 = 1 + x4
    tmp25 = tmp24 >= tmp2
    tmp26 = tmp24 < tmp4
    tmp27 = tmp25 & tmp26
    tmp28 = tmp6 & tmp27
    tmp29 = tl.load(in_ptr0 + ((-16) + x2 + (289*y3)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tmp31 + tmp23
    tmp33 = x5
    tmp34 = tmp33 >= tmp2
    tmp35 = tmp33 < tmp4
    tmp36 = tmp34 & tmp35
    tmp37 = tmp36 & tmp10
    tmp38 = tl.load(in_ptr0 + ((-1) + x2 + (289*y3)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tmp40 + tmp32
    tmp42 = tmp36 & tmp18
    tmp43 = tl.load(in_ptr0 + (x2 + (289*y3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp36 & tmp27
    tmp48 = tl.load(in_ptr0 + (1 + x2 + (289*y3)), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tmp50 + tmp46
    tmp52 = 1 + x5
    tmp53 = tmp52 >= tmp2
    tmp54 = tmp52 < tmp4
    tmp55 = tmp53 & tmp54
    tmp56 = tmp55 & tmp10
    tmp57 = tl.load(in_ptr0 + (16 + x2 + (289*y3)), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp56, tmp57, tmp58)
    tmp60 = tmp59 + tmp51
    tmp61 = tmp55 & tmp18
    tmp62 = tl.load(in_ptr0 + (17 + x2 + (289*y3)), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp61, tmp62, tmp63)
    tmp65 = tmp64 + tmp60
    tmp66 = tmp55 & tmp27
    tmp67 = tl.load(in_ptr0 + (18 + x2 + (289*y3)), tmp66 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp66, tmp67, tmp68)
    tmp70 = tmp69 + tmp65
    tmp71 = tl.full([1, 1], -1, tl.int64)
    tmp72 = tmp1 >= tmp71
    tmp73 = tl.full([1, 1], 18, tl.int64)
    tmp74 = tmp1 < tmp73
    tmp75 = tmp72 & tmp74
    tmp76 = tmp7 >= tmp71
    tmp77 = tmp7 < tmp73
    tmp78 = tmp76 & tmp77
    tmp79 = tmp75 & tmp78
    tmp80 = tl.broadcast_to((-1) + x5, [XBLOCK, YBLOCK])
    tmp81 = tmp80 >= tmp2
    tmp82 = tmp80 < tmp4
    tmp83 = tmp81 & tmp82
    tmp84 = tl.broadcast_to((-1) + x4, [XBLOCK, YBLOCK])
    tmp85 = tmp84 >= tmp2
    tmp86 = tmp84 < tmp4
    tmp87 = tmp85 & tmp86
    tmp88 = tmp83 & tmp87
    tmp89 = tmp88 & tmp79
    tmp90 = 1.0
    tmp91 = tl.full(tmp90.shape, 1.0, tmp90.dtype)
    tmp92 = tl.where(tmp89, tmp90, tmp91)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp79, tmp92, tmp93)
    tmp95 = tmp15 >= tmp71
    tmp96 = tmp15 < tmp73
    tmp97 = tmp95 & tmp96
    tmp98 = tmp75 & tmp97
    tmp99 = tl.broadcast_to(x4, [XBLOCK, YBLOCK])
    tmp100 = tmp99 >= tmp2
    tmp101 = tmp99 < tmp4
    tmp102 = tmp100 & tmp101
    tmp103 = tmp83 & tmp102
    tmp104 = tmp103 & tmp98
    tmp105 = tl.where(tmp104, tmp90, tmp91)
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp98, tmp105, tmp106)
    tmp108 = tmp107 + tmp94
    tmp109 = tmp24 >= tmp71
    tmp110 = tmp24 < tmp73
    tmp111 = tmp109 & tmp110
    tmp112 = tmp75 & tmp111
    tmp113 = tl.broadcast_to(1 + x4, [XBLOCK, YBLOCK])
    tmp114 = tmp113 >= tmp2
    tmp115 = tmp113 < tmp4
    tmp116 = tmp114 & tmp115
    tmp117 = tmp83 & tmp116
    tmp118 = tmp117 & tmp112
    tmp119 = tl.where(tmp118, tmp90, tmp91)
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp112, tmp119, tmp120)
    tmp122 = tmp121 + tmp108
    tmp123 = tmp33 >= tmp71
    tmp124 = tmp33 < tmp73
    tmp125 = tmp123 & tmp124
    tmp126 = tmp125 & tmp78
    tmp127 = tl.broadcast_to(x5, [XBLOCK, YBLOCK])
    tmp128 = tmp127 >= tmp2
    tmp129 = tmp127 < tmp4
    tmp130 = tmp128 & tmp129
    tmp131 = tmp130 & tmp87
    tmp132 = tmp131 & tmp126
    tmp133 = tl.where(tmp132, tmp90, tmp91)
    tmp134 = tl.full(tmp133.shape, 0.0, tmp133.dtype)
    tmp135 = tl.where(tmp126, tmp133, tmp134)
    tmp136 = tmp135 + tmp122
    tmp137 = tmp125 & tmp97
    tmp138 = tmp130 & tmp102
    tmp139 = tmp138 & tmp137
    tmp140 = tl.where(tmp139, tmp90, tmp91)
    tmp141 = tl.full(tmp140.shape, 0.0, tmp140.dtype)
    tmp142 = tl.where(tmp137, tmp140, tmp141)
    tmp143 = tmp142 + tmp136
    tmp144 = tmp125 & tmp111
    tmp145 = tmp130 & tmp116
    tmp146 = tmp145 & tmp144
    tmp147 = tl.where(tmp146, tmp90, tmp91)
    tmp148 = tl.full(tmp147.shape, 0.0, tmp147.dtype)
    tmp149 = tl.where(tmp144, tmp147, tmp148)
    tmp150 = tmp149 + tmp143
    tmp151 = tmp52 >= tmp71
    tmp152 = tmp52 < tmp73
    tmp153 = tmp151 & tmp152
    tmp154 = tmp153 & tmp78
    tmp155 = tl.broadcast_to(1 + x5, [XBLOCK, YBLOCK])
    tmp156 = tmp155 >= tmp2
    tmp157 = tmp155 < tmp4
    tmp158 = tmp156 & tmp157
    tmp159 = tmp158 & tmp87
    tmp160 = tmp159 & tmp154
    tmp161 = tl.where(tmp160, tmp90, tmp91)
    tmp162 = tl.full(tmp161.shape, 0.0, tmp161.dtype)
    tmp163 = tl.where(tmp154, tmp161, tmp162)
    tmp164 = tmp163 + tmp150
    tmp165 = tmp153 & tmp97
    tmp166 = tmp158 & tmp102
    tmp167 = tmp166 & tmp165
    tmp168 = tl.where(tmp167, tmp90, tmp91)
    tmp169 = tl.full(tmp168.shape, 0.0, tmp168.dtype)
    tmp170 = tl.where(tmp165, tmp168, tmp169)
    tmp171 = tmp170 + tmp164
    tmp172 = tmp153 & tmp111
    tmp173 = tmp158 & tmp116
    tmp174 = tmp173 & tmp172
    tmp175 = tl.where(tmp174, tmp90, tmp91)
    tmp176 = tl.full(tmp175.shape, 0.0, tmp175.dtype)
    tmp177 = tl.where(tmp172, tmp175, tmp176)
    tmp178 = tmp177 + tmp171
    tmp179 = tmp70 / tmp178
    tl.store(out_ptr0 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + (768*x2) + (221952*y1)), tmp179, xmask)
