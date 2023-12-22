
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: 'i32', 27: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(26, 27))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, out_ptr1, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    r2 = rindex % 12
    r3 = (rindex // 12)
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (144*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr6 + (r1 + (144*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr7 + (r1 + (144*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr9 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp32 = tl.load(in_ptr10 + (r1 + (144*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr12 + (0))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp43 = tl.load(in_ptr13 + (r1 + (144*x0)), rmask, other=0.0)
    tmp44 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr15 + (0))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp52 = tl.load(in_ptr16 + (r1 + (144*x0)), rmask, other=0.0)
    tmp53 = tl.load(in_ptr17 + (x0), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr18 + (0))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp64 = tl.load(in_ptr19 + ((6*(tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(6, 1 + (r3 // 2)))))) + (6*(tl.where((tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(6, 1 + (r3 // 2))))) >= 0, 0, 6))) + (36*x0) + (tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(6, 1 + (r2 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(6, 1 + (r2 // 2))))) >= 0, 0, 6))), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp23 * tmp25
    tmp27 = tmp26 * tmp4
    tmp30 = tmp27 * tmp29
    tmp31 = tmp30 * tmp9
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = tmp32 * tmp34
    tmp36 = tmp35 * tmp4
    tmp39 = tmp36 * tmp38
    tmp40 = tmp39 * tmp9
    tmp41 = tmp40 + tmp22
    tmp42 = tmp31 + tmp41
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp43 * tmp45
    tmp47 = tmp46 * tmp4
    tmp50 = tmp47 * tmp49
    tmp51 = tmp50 * tmp9
    tmp54 = tl.sigmoid(tmp53)
    tmp55 = tmp52 * tmp54
    tmp56 = tmp55 * tmp4
    tmp59 = tmp56 * tmp58
    tmp60 = tmp59 * tmp9
    tmp61 = tmp60 + tmp42
    tmp62 = tmp51 + tmp61
    tmp65 = tmp64 / 4
    tmp66 = tl.math.max(0, (r3 // 2))
    tmp67 = tl.math.min(6, 1 + (r3 // 2))
    tmp68 = tmp66 < tmp67
    tmp69 = tl.math.max(0, (r2 // 2))
    tmp70 = tl.math.min(6, 1 + (r2 // 2))
    tmp71 = tmp69 < tmp70
    tmp72 = tmp68 & tmp71
    tmp73 = 0.0
    tmp74 = tl.where(tmp72, tmp65, tmp73)
    tmp75 = tmp63 + tmp74
    tmp76 = 0.8980265101338745
    tmp77 = tmp75 * tmp76
    tmp78 = 1.7015043497085571
    tmp79 = tmp77 * tmp78
    tmp80 = 0.7071067811865476
    tmp81 = tmp62 * tmp80
    tmp82 = tl.math.erf(tmp81)
    tmp83 = 1.0
    tmp84 = tmp82 + tmp83
    tmp85 = 0.5
    tmp86 = tmp84 * tmp85
    tmp87 = tmp62 * tmp62
    tmp88 = -0.5
    tmp89 = tmp87 * tmp88
    tmp90 = tl.exp(tmp89)
    tmp91 = 0.3989422804014327
    tmp92 = tmp90 * tmp91
    tmp93 = tmp62 * tmp92
    tmp94 = tmp86 + tmp93
    tmp95 = tmp79 * tmp94
    tmp96 = tmp61 * tmp80
    tmp97 = tl.math.erf(tmp96)
    tmp98 = tmp97 + tmp83
    tmp99 = tmp98 * tmp85
    tmp100 = tmp61 * tmp61
    tmp101 = tmp100 * tmp88
    tmp102 = tl.exp(tmp101)
    tmp103 = tmp102 * tmp91
    tmp104 = tmp61 * tmp103
    tmp105 = tmp99 + tmp104
    tmp106 = tmp41 * tmp80
    tmp107 = tl.math.erf(tmp106)
    tmp108 = tmp107 + tmp83
    tmp109 = tmp108 * tmp85
    tmp110 = tmp41 * tmp41
    tmp111 = tmp110 * tmp88
    tmp112 = tl.exp(tmp111)
    tmp113 = tmp112 * tmp91
    tmp114 = tmp41 * tmp113
    tmp115 = tmp109 + tmp114
    tmp116 = tmp95 * tmp9
    tmp117 = tmp116 * tmp49
    tmp118 = tmp117 * tmp4
    tmp119 = tmp118 * tmp43
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK, RBLOCK])
    tmp122 = tl.where(rmask, tmp120, 0)
    tmp123 = tl.sum(tmp122, 1)[:, None]
    tmp124 = tmp83 - tmp45
    tmp125 = tmp45 * tmp124
    tmp126 = tmp123 * tmp125
    tl.store(out_ptr0 + (r1 + (144*x0)), tmp22, rmask)
    tl.store(out_ptr1 + (r1 + (144*x0)), tmp42, rmask)
    tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp95, rmask)
    tl.store(out_ptr3 + (r1 + (144*x0)), tmp105, rmask)
    tl.store(out_ptr4 + (r1 + (144*x0)), tmp115, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp126, None)
