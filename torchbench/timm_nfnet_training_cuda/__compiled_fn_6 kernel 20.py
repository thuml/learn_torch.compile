
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (144*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr4 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 0.7071067811865476
    tmp14 = tmp12 * tmp13
    tmp15 = tl.math.erf(tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.5
    tmp19 = tmp17 * tmp18
    tmp20 = tmp12 * tmp12
    tmp21 = -0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tl.exp(tmp22)
    tmp24 = 0.3989422804014327
    tmp25 = tmp23 * tmp24
    tmp26 = tmp12 * tmp25
    tmp27 = tmp19 + tmp26
    tmp30 = 0.9805806756909201
    tmp31 = tmp29 * tmp30
    tmp32 = 1.7015043497085571
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33 * tmp27
    tmp35 = tmp28 + tmp34
    tmp36 = tmp35 * tmp9
    tmp37 = tmp36 * tmp7
    tmp38 = tmp37 * tmp4
    tmp39 = tmp38 * tmp0
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(rmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp44 = tmp16 - tmp2
    tmp45 = tmp2 * tmp44
    tmp46 = tmp43 * tmp45
    tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp35, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp46, None)
