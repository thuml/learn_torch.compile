
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20, 21))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (r1 + (49*x0) + (50176*r2)), rmask & xmask).to(tl.int1)
    tmp22 = tl.load(in_ptr5 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tl.where(tmp16, tmp2, tmp1)
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp24 = tmp22 - tmp23
    tmp25 = tmp17 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp32 = tmp30 - tmp31
    tmp33 = tmp17 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp39 = tmp29 * tmp38
    tmp41 = tmp37 * tmp40
    tmp43 = tmp15 * tmp42
    tl.store(out_ptr5 + (x0), tmp39, xmask)
    tl.store(out_ptr6 + (x0), tmp41, xmask)
    tl.store(out_ptr7 + (x0), tmp43, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr3 + (x0), tmp29, xmask)
    tl.store(out_ptr4 + (x0), tmp37, xmask)
