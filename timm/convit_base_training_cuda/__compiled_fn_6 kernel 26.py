
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x1 = (xindex // 196) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 1.0
    tmp5 = tmp4 - tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = tmp3 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12 / tmp11
    tmp14 = tmp1 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp0 / tmp11
    tmp20 = tmp19 + tmp18
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 * tmp8
    tmp23 = tmp20 * tmp5
    tmp24 = tmp23 * tmp6
    tmp25 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp6 * tmp32
    tmp34 = tmp24 - tmp33
    tmp35 = 0.14433756729740643
    tmp36 = tmp34 * tmp35
    tl.store(out_ptr1 + (r3 + (196*x4)), tmp22, rmask & xmask)
    tl.store(out_ptr5 + (r3 + (196*x4)), tmp36, rmask & xmask)
    tl.store(out_ptr0 + (x4), tmp18, xmask)
    tl.store(out_ptr3 + (x4), tmp28, xmask)
