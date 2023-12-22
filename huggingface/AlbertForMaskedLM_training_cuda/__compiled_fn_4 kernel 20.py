
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: 'i32', 39: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(38, 39))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp26 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp39 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp42 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp52 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp53 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp65 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp66 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp68 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp78 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp79 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp81 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp91 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp92 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp94 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp104 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp105 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp107 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp117 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp118 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp120 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp130 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp131 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp133 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp143 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp144 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp146 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp41 = tmp39 + tmp40
    tmp43 = tmp41 * tmp42
    tmp44 = tl.broadcast_to(tmp43, [RBLOCK])
    tmp46 = tl.where(rmask, tmp44, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 0))
    tmp48 = tl.broadcast_to(tmp41, [RBLOCK])
    tmp50 = tl.where(rmask, tmp48, 0)
    tmp51 = triton_helpers.promote_to_tensor(tl.sum(tmp50, 0))
    tmp54 = tmp52 + tmp53
    tmp56 = tmp54 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp59 = tl.where(rmask, tmp57, 0)
    tmp60 = triton_helpers.promote_to_tensor(tl.sum(tmp59, 0))
    tmp61 = tl.broadcast_to(tmp54, [RBLOCK])
    tmp63 = tl.where(rmask, tmp61, 0)
    tmp64 = triton_helpers.promote_to_tensor(tl.sum(tmp63, 0))
    tmp67 = tmp65 + tmp66
    tmp69 = tmp67 * tmp68
    tmp70 = tl.broadcast_to(tmp69, [RBLOCK])
    tmp72 = tl.where(rmask, tmp70, 0)
    tmp73 = triton_helpers.promote_to_tensor(tl.sum(tmp72, 0))
    tmp74 = tl.broadcast_to(tmp67, [RBLOCK])
    tmp76 = tl.where(rmask, tmp74, 0)
    tmp77 = triton_helpers.promote_to_tensor(tl.sum(tmp76, 0))
    tmp80 = tmp78 + tmp79
    tmp82 = tmp80 * tmp81
    tmp83 = tl.broadcast_to(tmp82, [RBLOCK])
    tmp85 = tl.where(rmask, tmp83, 0)
    tmp86 = triton_helpers.promote_to_tensor(tl.sum(tmp85, 0))
    tmp87 = tl.broadcast_to(tmp80, [RBLOCK])
    tmp89 = tl.where(rmask, tmp87, 0)
    tmp90 = triton_helpers.promote_to_tensor(tl.sum(tmp89, 0))
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tl.broadcast_to(tmp95, [RBLOCK])
    tmp98 = tl.where(rmask, tmp96, 0)
    tmp99 = triton_helpers.promote_to_tensor(tl.sum(tmp98, 0))
    tmp100 = tl.broadcast_to(tmp93, [RBLOCK])
    tmp102 = tl.where(rmask, tmp100, 0)
    tmp103 = triton_helpers.promote_to_tensor(tl.sum(tmp102, 0))
    tmp106 = tmp104 + tmp105
    tmp108 = tmp106 * tmp107
    tmp109 = tl.broadcast_to(tmp108, [RBLOCK])
    tmp111 = tl.where(rmask, tmp109, 0)
    tmp112 = triton_helpers.promote_to_tensor(tl.sum(tmp111, 0))
    tmp113 = tl.broadcast_to(tmp106, [RBLOCK])
    tmp115 = tl.where(rmask, tmp113, 0)
    tmp116 = triton_helpers.promote_to_tensor(tl.sum(tmp115, 0))
    tmp119 = tmp117 + tmp118
    tmp121 = tmp119 * tmp120
    tmp122 = tl.broadcast_to(tmp121, [RBLOCK])
    tmp124 = tl.where(rmask, tmp122, 0)
    tmp125 = triton_helpers.promote_to_tensor(tl.sum(tmp124, 0))
    tmp126 = tl.broadcast_to(tmp119, [RBLOCK])
    tmp128 = tl.where(rmask, tmp126, 0)
    tmp129 = triton_helpers.promote_to_tensor(tl.sum(tmp128, 0))
    tmp132 = tmp130 + tmp131
    tmp134 = tmp132 * tmp133
    tmp135 = tl.broadcast_to(tmp134, [RBLOCK])
    tmp137 = tl.where(rmask, tmp135, 0)
    tmp138 = triton_helpers.promote_to_tensor(tl.sum(tmp137, 0))
    tmp139 = tl.broadcast_to(tmp132, [RBLOCK])
    tmp141 = tl.where(rmask, tmp139, 0)
    tmp142 = triton_helpers.promote_to_tensor(tl.sum(tmp141, 0))
    tmp145 = tmp143 + tmp144
    tmp147 = tmp145 * tmp146
    tmp148 = tl.broadcast_to(tmp147, [RBLOCK])
    tmp150 = tl.where(rmask, tmp148, 0)
    tmp151 = triton_helpers.promote_to_tensor(tl.sum(tmp150, 0))
    tmp152 = tl.broadcast_to(tmp145, [RBLOCK])
    tmp154 = tl.where(rmask, tmp152, 0)
    tmp155 = triton_helpers.promote_to_tensor(tl.sum(tmp154, 0))
    tmp156 = tmp8 + tmp21
    tmp157 = tmp156 + tmp34
    tmp158 = tmp157 + tmp47
    tmp159 = tmp158 + tmp60
    tmp160 = tmp159 + tmp73
    tmp161 = tmp160 + tmp99
    tmp162 = tmp161 + tmp125
    tmp163 = tmp162 + tmp151
    tmp164 = tmp163 + tmp86
    tmp165 = tmp164 + tmp112
    tmp166 = tmp165 + tmp138
    tmp167 = tmp12 + tmp25
    tmp168 = tmp167 + tmp38
    tmp169 = tmp168 + tmp51
    tmp170 = tmp169 + tmp64
    tmp171 = tmp170 + tmp77
    tmp172 = tmp171 + tmp103
    tmp173 = tmp172 + tmp129
    tmp174 = tmp173 + tmp155
    tmp175 = tmp174 + tmp90
    tmp176 = tmp175 + tmp116
    tmp177 = tmp176 + tmp142
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp166, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp177, None)
