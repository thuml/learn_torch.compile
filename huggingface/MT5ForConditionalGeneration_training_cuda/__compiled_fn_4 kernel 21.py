
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*i1', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*i1', 22: '*fp32', 23: '*fp32', 24: 'i32', 25: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(24, 25))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp38 = tl.load(in_ptr17 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr18 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp46 = tl.load(in_ptr19 + (x0), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 1.1111111111111112
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 * tmp40
    tmp42 = tl.broadcast_to(tmp41, [RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = triton_helpers.promote_to_tensor(tl.sum(tmp44, 0))
    tmp47 = tmp39 * tmp46
    tmp48 = -0.5
    tmp49 = tmp45 * tmp48
    tmp50 = tmp46 * tmp46
    tmp51 = tmp50 * tmp46
    tmp52 = tmp49 * tmp51
    tmp53 = 512.0
    tmp54 = tmp52 / tmp53
    tmp55 = 2.0
    tmp56 = tmp40 * tmp55
    tmp57 = tmp54 * tmp56
    tmp58 = tmp47 + tmp57
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 * tmp35
    tmp62 = tmp58 * tmp61
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp58, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp62, rmask & xmask)
