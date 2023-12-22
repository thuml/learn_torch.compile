
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 14)
    x2 = xindex % 14
    y0 = yindex % 56
    y1 = (yindex // 56)
    x5 = xindex
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x2)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-12600) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-12152) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-11704) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-56) + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (392 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (840 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (12488 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (12936 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (13384 + y0 + (896*x2) + (25088*x3) + (351232*y1)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1, 1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1, 1], 29, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tl.broadcast_to((-1) + (2*x3), [XBLOCK, YBLOCK])
    tmp80 = tmp79 >= tmp1
    tmp81 = tmp79 < tmp3
    tmp82 = tmp80 & tmp81
    tmp83 = tl.broadcast_to((-1) + (2*x2), [XBLOCK, YBLOCK])
    tmp84 = tmp83 >= tmp1
    tmp85 = tmp83 < tmp3
    tmp86 = tmp84 & tmp85
    tmp87 = tmp82 & tmp86
    tmp88 = tmp87 & tmp78
    tmp89 = 1.0
    tmp90 = tl.full(tmp89.shape, 1.0, tmp89.dtype)
    tmp91 = tl.where(tmp88, tmp89, tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp78, tmp91, tmp92)
    tmp94 = tmp14 >= tmp70
    tmp95 = tmp14 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tl.broadcast_to(2*x2, [XBLOCK, YBLOCK])
    tmp99 = tmp98 >= tmp1
    tmp100 = tmp98 < tmp3
    tmp101 = tmp99 & tmp100
    tmp102 = tmp82 & tmp101
    tmp103 = tmp102 & tmp97
    tmp104 = tl.where(tmp103, tmp89, tmp90)
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp97, tmp104, tmp105)
    tmp107 = tmp106 + tmp93
    tmp108 = tmp23 >= tmp70
    tmp109 = tmp23 < tmp72
    tmp110 = tmp108 & tmp109
    tmp111 = tmp74 & tmp110
    tmp112 = tl.broadcast_to(1 + (2*x2), [XBLOCK, YBLOCK])
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp82 & tmp115
    tmp117 = tmp116 & tmp111
    tmp118 = tl.where(tmp117, tmp89, tmp90)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp111, tmp118, tmp119)
    tmp121 = tmp120 + tmp107
    tmp122 = tmp32 >= tmp70
    tmp123 = tmp32 < tmp72
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp77
    tmp126 = tl.broadcast_to(2*x3, [XBLOCK, YBLOCK])
    tmp127 = tmp126 >= tmp1
    tmp128 = tmp126 < tmp3
    tmp129 = tmp127 & tmp128
    tmp130 = tmp129 & tmp86
    tmp131 = tmp130 & tmp125
    tmp132 = tl.where(tmp131, tmp89, tmp90)
    tmp133 = tl.full(tmp132.shape, 0.0, tmp132.dtype)
    tmp134 = tl.where(tmp125, tmp132, tmp133)
    tmp135 = tmp134 + tmp121
    tmp136 = tmp124 & tmp96
    tmp137 = tmp129 & tmp101
    tmp138 = tmp137 & tmp136
    tmp139 = tl.where(tmp138, tmp89, tmp90)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tmp141 + tmp135
    tmp143 = tmp124 & tmp110
    tmp144 = tmp129 & tmp115
    tmp145 = tmp144 & tmp143
    tmp146 = tl.where(tmp145, tmp89, tmp90)
    tmp147 = tl.full(tmp146.shape, 0.0, tmp146.dtype)
    tmp148 = tl.where(tmp143, tmp146, tmp147)
    tmp149 = tmp148 + tmp142
    tmp150 = tmp51 >= tmp70
    tmp151 = tmp51 < tmp72
    tmp152 = tmp150 & tmp151
    tmp153 = tmp152 & tmp77
    tmp154 = tl.broadcast_to(1 + (2*x3), [XBLOCK, YBLOCK])
    tmp155 = tmp154 >= tmp1
    tmp156 = tmp154 < tmp3
    tmp157 = tmp155 & tmp156
    tmp158 = tmp157 & tmp86
    tmp159 = tmp158 & tmp153
    tmp160 = tl.where(tmp159, tmp89, tmp90)
    tmp161 = tl.full(tmp160.shape, 0.0, tmp160.dtype)
    tmp162 = tl.where(tmp153, tmp160, tmp161)
    tmp163 = tmp162 + tmp149
    tmp164 = tmp152 & tmp96
    tmp165 = tmp157 & tmp101
    tmp166 = tmp165 & tmp164
    tmp167 = tl.where(tmp166, tmp89, tmp90)
    tmp168 = tl.full(tmp167.shape, 0.0, tmp167.dtype)
    tmp169 = tl.where(tmp164, tmp167, tmp168)
    tmp170 = tmp169 + tmp163
    tmp171 = tmp152 & tmp110
    tmp172 = tmp157 & tmp115
    tmp173 = tmp172 & tmp171
    tmp174 = tl.where(tmp173, tmp89, tmp90)
    tmp175 = tl.full(tmp174.shape, 0.0, tmp174.dtype)
    tmp176 = tl.where(tmp171, tmp174, tmp175)
    tmp177 = tmp176 + tmp170
    tmp178 = tmp69 / tmp177
    tl.store(out_ptr0 + (x5 + (196*y0) + (87808*y1)), tmp178, xmask & ymask)
