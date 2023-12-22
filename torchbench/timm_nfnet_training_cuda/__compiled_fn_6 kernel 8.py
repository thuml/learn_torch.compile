
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.load(in_ptr3 + (r1 + (36*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr4 + (r1 + (36*x0)), rmask, other=0.0)
    tmp29 = tl.load(in_ptr5 + (r1 + (36*x0)), rmask, other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1 + (36*x0)), rmask, other=0.0)
    tmp47 = tl.load(in_out_ptr0 + (r1 + (36*x0)), rmask, other=0.0)
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
    tmp30 = 0.9622504486493761
    tmp31 = tmp29 * tmp30
    tmp32 = 1.7015043497085571
    tmp33 = tmp31 * tmp32
    tmp35 = tmp34 * tmp13
    tmp36 = tl.math.erf(tmp35)
    tmp37 = tmp36 + tmp16
    tmp38 = tmp37 * tmp18
    tmp39 = tmp34 * tmp34
    tmp40 = tmp39 * tmp21
    tmp41 = tl.exp(tmp40)
    tmp42 = tmp41 * tmp24
    tmp43 = tmp34 * tmp42
    tmp44 = tmp38 + tmp43
    tmp45 = tmp33 * tmp44
    tmp46 = tmp28 + tmp45
    tmp48 = 0.9805806756909201
    tmp49 = tmp47 * tmp48
    tmp50 = tmp49 * tmp32
    tmp51 = tmp50 * tmp27
    tmp52 = tmp46 + tmp51
    tmp53 = tmp52 * tmp9
    tmp54 = tmp53 * tmp7
    tmp55 = tmp54 * tmp4
    tmp56 = tmp55 * tmp0
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.where(rmask, tmp57, 0)
    tmp60 = tl.sum(tmp59, 1)[:, None]
    tmp61 = tmp16 - tmp2
    tmp62 = tmp2 * tmp61
    tmp63 = tmp60 * tmp62
    tl.store(in_out_ptr0 + (r1 + (36*x0)), tmp52, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp63, None)
