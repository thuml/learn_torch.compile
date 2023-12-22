
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: 'i32', 28: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(27, 28))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr5, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask, other=0.0)
    tmp31 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr17 + (r1 + (512*x0)), rmask, other=0.0)
    tmp36 = tl.load(in_ptr18 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr19 + (r1 + (512*x0)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask, other=0.0)
    tmp47 = tl.load(in_ptr21 + (x0), None, eviction_policy='evict_last')
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
    tmp34 = 0.04419417382415922
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp32 + tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp37 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [RBLOCK])
    tmp45 = tl.where(rmask, tmp43, 0)
    tmp46 = triton_helpers.promote_to_tensor(tl.sum(tmp45, 0))
    tmp48 = tmp37 * tmp47
    tmp49 = -0.5
    tmp50 = tmp46 * tmp49
    tmp51 = tmp47 * tmp47
    tmp52 = tmp51 * tmp47
    tmp53 = tmp50 * tmp52
    tmp54 = 512.0
    tmp55 = tmp53 / tmp54
    tmp56 = 2.0
    tmp57 = tmp41 * tmp56
    tmp58 = tmp55 * tmp57
    tmp59 = tmp48 + tmp58
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp16, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp32, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp59, rmask)
