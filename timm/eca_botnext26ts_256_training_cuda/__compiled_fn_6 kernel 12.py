
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_batch_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = (rindex // 64)
    x0 = xindex
    r1 = rindex % 64
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 - tmp14
    tmp16 = tmp8 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp21 - tmp22
    tmp24 = tmp8 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp20 * tmp29
    tmp32 = tmp28 * tmp31
    tl.store(out_ptr3 + (x0), tmp30, None)
    tl.store(out_ptr4 + (x0), tmp32, None)
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp20, None)
    tl.store(out_ptr2 + (x0), tmp28, None)
