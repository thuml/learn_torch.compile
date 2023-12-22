
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_bitwise_not_detach_lift_fresh_masked_fill_mul_rsub_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 6144
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.full([1], False, tl.int1)
    tmp2 = -3.4028234663852886e+38
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = (tmp16 != 0)
    tmp18 = tmp9 / tmp13
    tmp19 = 0.0
    tmp20 = tl.where(tmp1, tmp19, tmp18)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp17, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp23, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp20, rmask)
