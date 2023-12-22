
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[262144, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 12)
    x0 = xindex % 12
    x1 = (xindex // 12) % 8
    x2 = (xindex // 96) % 8
    x3 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r4 + (12*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (12*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.25
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (12*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (12*x2) + (96*x1) + (768*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
