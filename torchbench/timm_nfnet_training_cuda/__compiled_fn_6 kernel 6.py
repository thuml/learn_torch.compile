
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_mul_sigmoid_sigmoid_backward_sum_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr6 + (r1 + (36*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr7 + (r1 + (36*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr8 + (r1 + (36*x0)), rmask, other=0.0)
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
    tmp25 = 0.9622504486493761
    tmp26 = tmp24 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.7071067811865476
    tmp30 = tmp22 * tmp29
    tmp31 = tl.math.erf(tmp30)
    tmp32 = 1.0
    tmp33 = tmp31 + tmp32
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = tmp22 * tmp22
    tmp37 = -0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tl.exp(tmp38)
    tmp40 = 0.3989422804014327
    tmp41 = tmp39 * tmp40
    tmp42 = tmp22 * tmp41
    tmp43 = tmp35 + tmp42
    tmp44 = tmp28 * tmp43
    tmp45 = tmp23 + tmp44
    tmp46 = tmp45 * tmp9
    tmp47 = tmp46 * tmp7
    tmp48 = tmp47 * tmp4
    tmp49 = tmp48 * tmp0
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
    tmp52 = tl.where(rmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp54 = tmp32 - tmp2
    tmp55 = tmp2 * tmp54
    tmp56 = tmp53 * tmp55
    tl.store(out_ptr0 + (r1 + (36*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp56, None)
