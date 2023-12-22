
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 196) % 16
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.14433756729740643
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = 1.0
    tmp17 = tmp16 - tmp15
    tmp18 = tmp17 * tmp13
    tmp20 = tmp15 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp21 / tmp25
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (196*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp25, xmask)
