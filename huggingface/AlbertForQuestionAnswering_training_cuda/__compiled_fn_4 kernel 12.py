
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: 'i32', 60: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(59, 60))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp31 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp35 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp45 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp46 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp48 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp50 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp52 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp62 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp63 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp65 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp67 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp69 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp79 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp80 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp82 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp84 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp86 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp96 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp97 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp99 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp101 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp103 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp113 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp114 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp116 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp118 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp120 = tl.load(in_ptr36 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp130 = tl.load(in_ptr37 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp131 = tl.load(in_ptr38 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp133 = tl.load(in_ptr39 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp135 = tl.load(in_ptr40 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp137 = tl.load(in_ptr41 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp147 = tl.load(in_ptr42 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp148 = tl.load(in_ptr43 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp150 = tl.load(in_ptr44 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp152 = tl.load(in_ptr45 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp154 = tl.load(in_ptr46 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp164 = tl.load(in_ptr47 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp165 = tl.load(in_ptr48 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp167 = tl.load(in_ptr49 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp169 = tl.load(in_ptr50 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp171 = tl.load(in_ptr51 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp181 = tl.load(in_ptr52 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp182 = tl.load(in_ptr53 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp184 = tl.load(in_ptr54 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp186 = tl.load(in_ptr55 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp188 = tl.load(in_ptr56 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, 0)
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tmp41 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp53 = tmp51 * tmp52
    tmp54 = tl.broadcast_to(tmp53, [RBLOCK])
    tmp56 = tl.where(rmask, tmp54, 0)
    tmp57 = triton_helpers.promote_to_tensor(tl.sum(tmp56, 0))
    tmp58 = tl.broadcast_to(tmp51, [RBLOCK])
    tmp60 = tl.where(rmask, tmp58, 0)
    tmp61 = triton_helpers.promote_to_tensor(tl.sum(tmp60, 0))
    tmp64 = tmp62 + tmp63
    tmp66 = tmp64 + tmp65
    tmp68 = tmp66 + tmp67
    tmp70 = tmp68 * tmp69
    tmp71 = tl.broadcast_to(tmp70, [RBLOCK])
    tmp73 = tl.where(rmask, tmp71, 0)
    tmp74 = triton_helpers.promote_to_tensor(tl.sum(tmp73, 0))
    tmp75 = tl.broadcast_to(tmp68, [RBLOCK])
    tmp77 = tl.where(rmask, tmp75, 0)
    tmp78 = triton_helpers.promote_to_tensor(tl.sum(tmp77, 0))
    tmp81 = tmp79 + tmp80
    tmp83 = tmp81 + tmp82
    tmp85 = tmp83 + tmp84
    tmp87 = tmp85 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [RBLOCK])
    tmp90 = tl.where(rmask, tmp88, 0)
    tmp91 = triton_helpers.promote_to_tensor(tl.sum(tmp90, 0))
    tmp92 = tl.broadcast_to(tmp85, [RBLOCK])
    tmp94 = tl.where(rmask, tmp92, 0)
    tmp95 = triton_helpers.promote_to_tensor(tl.sum(tmp94, 0))
    tmp98 = tmp96 + tmp97
    tmp100 = tmp98 + tmp99
    tmp102 = tmp100 + tmp101
    tmp104 = tmp102 * tmp103
    tmp105 = tl.broadcast_to(tmp104, [RBLOCK])
    tmp107 = tl.where(rmask, tmp105, 0)
    tmp108 = triton_helpers.promote_to_tensor(tl.sum(tmp107, 0))
    tmp109 = tl.broadcast_to(tmp102, [RBLOCK])
    tmp111 = tl.where(rmask, tmp109, 0)
    tmp112 = triton_helpers.promote_to_tensor(tl.sum(tmp111, 0))
    tmp115 = tmp113 + tmp114
    tmp117 = tmp115 + tmp116
    tmp119 = tmp117 + tmp118
    tmp121 = tmp119 * tmp120
    tmp122 = tl.broadcast_to(tmp121, [RBLOCK])
    tmp124 = tl.where(rmask, tmp122, 0)
    tmp125 = triton_helpers.promote_to_tensor(tl.sum(tmp124, 0))
    tmp126 = tl.broadcast_to(tmp119, [RBLOCK])
    tmp128 = tl.where(rmask, tmp126, 0)
    tmp129 = triton_helpers.promote_to_tensor(tl.sum(tmp128, 0))
    tmp132 = tmp130 + tmp131
    tmp134 = tmp132 + tmp133
    tmp136 = tmp134 + tmp135
    tmp138 = tmp136 * tmp137
    tmp139 = tl.broadcast_to(tmp138, [RBLOCK])
    tmp141 = tl.where(rmask, tmp139, 0)
    tmp142 = triton_helpers.promote_to_tensor(tl.sum(tmp141, 0))
    tmp143 = tl.broadcast_to(tmp136, [RBLOCK])
    tmp145 = tl.where(rmask, tmp143, 0)
    tmp146 = triton_helpers.promote_to_tensor(tl.sum(tmp145, 0))
    tmp149 = tmp147 + tmp148
    tmp151 = tmp149 + tmp150
    tmp153 = tmp151 + tmp152
    tmp155 = tmp153 * tmp154
    tmp156 = tl.broadcast_to(tmp155, [RBLOCK])
    tmp158 = tl.where(rmask, tmp156, 0)
    tmp159 = triton_helpers.promote_to_tensor(tl.sum(tmp158, 0))
    tmp160 = tl.broadcast_to(tmp153, [RBLOCK])
    tmp162 = tl.where(rmask, tmp160, 0)
    tmp163 = triton_helpers.promote_to_tensor(tl.sum(tmp162, 0))
    tmp166 = tmp164 + tmp165
    tmp168 = tmp166 + tmp167
    tmp170 = tmp168 + tmp169
    tmp172 = tmp170 * tmp171
    tmp173 = tl.broadcast_to(tmp172, [RBLOCK])
    tmp175 = tl.where(rmask, tmp173, 0)
    tmp176 = triton_helpers.promote_to_tensor(tl.sum(tmp175, 0))
    tmp177 = tl.broadcast_to(tmp170, [RBLOCK])
    tmp179 = tl.where(rmask, tmp177, 0)
    tmp180 = triton_helpers.promote_to_tensor(tl.sum(tmp179, 0))
    tmp183 = tmp181 + tmp182
    tmp185 = tmp183 + tmp184
    tmp187 = tmp185 + tmp186
    tmp189 = tmp187 * tmp188
    tmp190 = tl.broadcast_to(tmp189, [RBLOCK])
    tmp192 = tl.where(rmask, tmp190, 0)
    tmp193 = triton_helpers.promote_to_tensor(tl.sum(tmp192, 0))
    tmp194 = tl.broadcast_to(tmp187, [RBLOCK])
    tmp196 = tl.where(rmask, tmp194, 0)
    tmp197 = triton_helpers.promote_to_tensor(tl.sum(tmp196, 0))
    tmp198 = tmp6 + tmp23
    tmp199 = tmp198 + tmp40
    tmp200 = tmp199 + tmp57
    tmp201 = tmp200 + tmp74
    tmp202 = tmp201 + tmp91
    tmp203 = tmp202 + tmp125
    tmp204 = tmp203 + tmp159
    tmp205 = tmp204 + tmp193
    tmp206 = tmp205 + tmp108
    tmp207 = tmp206 + tmp142
    tmp208 = tmp207 + tmp176
    tmp209 = tmp10 + tmp27
    tmp210 = tmp209 + tmp44
    tmp211 = tmp210 + tmp61
    tmp212 = tmp211 + tmp78
    tmp213 = tmp212 + tmp95
    tmp214 = tmp213 + tmp129
    tmp215 = tmp214 + tmp163
    tmp216 = tmp215 + tmp197
    tmp217 = tmp216 + tmp112
    tmp218 = tmp217 + tmp146
    tmp219 = tmp218 + tmp180
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp208, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp219, None)
