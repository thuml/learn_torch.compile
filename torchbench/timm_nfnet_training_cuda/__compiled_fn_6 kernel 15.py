
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (144*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr4 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp38 = tl.load(in_ptr5 + (r1 + (144*x0)), rmask, other=0.0)
    tmp44 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9128709291752768
    tmp3 = tmp1 * tmp2
    tmp4 = 1.7015043497085571
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tmp10 = 0.9284766908852592
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp4
    tmp14 = 0.7071067811865476
    tmp15 = tmp13 * tmp14
    tmp16 = tl.math.erf(tmp15)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp13
    tmp22 = -0.5
    tmp23 = tmp21 * tmp22
    tmp24 = tl.exp(tmp23)
    tmp25 = 0.3989422804014327
    tmp26 = tmp24 * tmp25
    tmp27 = tmp13 * tmp26
    tmp28 = tmp20 + tmp27
    tmp29 = tmp12 * tmp28
    tmp30 = tmp8 + tmp29
    tmp31 = 0.2
    tmp32 = tmp30 * tmp31
    tmp35 = tmp32 * tmp34
    tmp36 = 2.0
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(rmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp17 - tmp45
    tmp47 = tmp45 * tmp46
    tmp48 = tmp43 * tmp47
    tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp30, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp48, None)
