
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = 1.0
    tmp20 = tmp19 - tmp18
    tmp21 = tmp18 * tmp20
    tmp22 = tmp16 * tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, None)
