
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp1 = 0.04419417382415922
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp9 * tmp16
    tmp18 = -0.5
    tmp19 = tmp15 * tmp18
    tmp20 = tmp16 * tmp16
    tmp21 = tmp20 * tmp16
    tmp22 = tmp19 * tmp21
    tmp23 = 512.0
    tmp24 = tmp22 / tmp23
    tmp25 = 2.0
    tmp26 = tmp10 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp17 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp5
    tmp32 = tmp28 * tmp31
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp28, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, rmask & xmask)
