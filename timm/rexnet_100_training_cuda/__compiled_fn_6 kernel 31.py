
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7248
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 906
    x1 = (xindex // 906)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (906*r2) + (44394*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp0
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = 1.0
    tmp17 = tmp16 - tmp2
    tmp18 = tmp2 * tmp17
    tmp19 = tmp15 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
