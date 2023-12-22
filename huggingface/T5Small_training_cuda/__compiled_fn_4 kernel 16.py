
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = -0.5
    tmp18 = tmp12 * tmp17
    tmp19 = tmp14 * tmp14
    tmp20 = tmp19 * tmp14
    tmp21 = tmp18 * tmp20
    tmp22 = 512.0
    tmp23 = tmp21 / tmp22
    tmp24 = 2.0
    tmp25 = tmp7 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp16 + tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 1.1111111111111112
    tmp31 = tmp29 * tmp30
    tmp32 = tmp27 * tmp31
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp32, rmask & xmask)
