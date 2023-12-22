
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr5, out_ptr6, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp42 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 / tmp10
    tmp12 = -tmp9
    tmp14 = tmp13 / tmp10
    tmp15 = tmp14 / tmp10
    tmp16 = tmp12 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 2.0
    tmp22 = tmp10 * tmp21
    tmp23 = tmp20 / tmp22
    tmp24 = 768.0
    tmp25 = tmp23 / tmp24
    tmp26 = tmp13 * tmp21
    tmp27 = tmp25 * tmp26
    tmp28 = -tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = -tmp11
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp11 + tmp27
    tmp39 = tmp37 + tmp32
    tmp40 = tmp39 / tmp24
    tmp41 = tmp38 + tmp40
    tmp43 = tl.full([1], -1, tl.int64)
    tmp44 = tmp42 == tmp43
    tmp45 = tl.where(tmp44, tmp4, tmp41)
    tmp47 = tl.full([1], 0, tl.int64)
    tmp48 = tmp46 == tmp47
    tmp49 = tl.where(tmp48, tmp4, tmp41)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp45, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp49, rmask & xmask)
