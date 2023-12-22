
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 174
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    r3 = rindex
    tmp22 = tl.load(in_ptr2 + (x0 + (174*r3)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 162, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp10 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp24 = tmp22 - tmp23
    tmp25 = tmp17 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr2 + (x0), tmp31, xmask)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (x0), tmp29, xmask)
