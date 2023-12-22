
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 112
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (112*r3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0 + (112*r3)), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp2 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp21 - tmp22
    tmp24 = tmp16 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp14 * tmp32
    tmp35 = tmp34 + tmp30
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp28 * tmp36
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp33, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
