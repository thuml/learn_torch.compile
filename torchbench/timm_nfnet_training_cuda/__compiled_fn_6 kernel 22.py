
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    r2 = rindex % 24
    r3 = (rindex // 24)
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (576*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp20 = tl.load(in_ptr6 + (r1 + (576*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr7 + (r1 + (576*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr8 + ((12*(tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(12, 1 + (r3 // 2)))))) + (12*(tl.where((tl.math.min(tl.math.max(0, (r3 // 2)), (-1) + (tl.math.min(12, 1 + (r3 // 2))))) >= 0, 0, 12))) + (144*x0) + (tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(12, 1 + (r2 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(12, 1 + (r2 // 2))))) >= 0, 0, 12))), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp25 = tmp24 / 4
    tmp26 = tl.math.max(0, (r3 // 2))
    tmp27 = tl.math.min(12, 1 + (r3 // 2))
    tmp28 = tmp26 < tmp27
    tmp29 = tl.math.max(0, (r2 // 2))
    tmp30 = tl.math.min(12, 1 + (r2 // 2))
    tmp31 = tmp29 < tmp30
    tmp32 = tmp28 & tmp31
    tmp33 = 0.0
    tmp34 = tl.where(tmp32, tmp25, tmp33)
    tmp35 = tmp23 + tmp34
    tmp36 = 0.9622504486493761
    tmp37 = tmp35 * tmp36
    tmp38 = 1.7015043497085571
    tmp39 = tmp37 * tmp38
    tmp40 = 0.7071067811865476
    tmp41 = tmp22 * tmp40
    tmp42 = tl.math.erf(tmp41)
    tmp43 = 1.0
    tmp44 = tmp42 + tmp43
    tmp45 = 0.5
    tmp46 = tmp44 * tmp45
    tmp47 = tmp22 * tmp22
    tmp48 = -0.5
    tmp49 = tmp47 * tmp48
    tmp50 = tl.exp(tmp49)
    tmp51 = 0.3989422804014327
    tmp52 = tmp50 * tmp51
    tmp53 = tmp22 * tmp52
    tmp54 = tmp46 + tmp53
    tmp55 = tmp39 * tmp54
    tmp56 = tmp55 * tmp9
    tmp57 = tmp56 * tmp7
    tmp58 = tmp57 * tmp4
    tmp59 = tmp58 * tmp0
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask, tmp60, 0)
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp62, 0))
    tmp64 = tmp43 - tmp2
    tmp65 = tmp2 * tmp64
    tmp66 = tmp63 * tmp65
    tl.store(in_out_ptr0 + (r1 + (576*x0)), tmp55, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp66, None)
