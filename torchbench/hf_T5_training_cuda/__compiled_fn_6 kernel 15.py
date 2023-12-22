
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp2 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = -0.5
    tmp20 = tmp14 * tmp19
    tmp21 = tmp16 * tmp16
    tmp22 = tmp21 * tmp16
    tmp23 = tmp20 * tmp22
    tmp24 = 512.0
    tmp25 = tmp23 / tmp24
    tmp26 = 2.0
    tmp27 = tmp9 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp18 + tmp28
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp29, rmask)
