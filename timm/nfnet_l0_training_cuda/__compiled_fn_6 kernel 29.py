
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sigmoid_sigmoid_backward_sum_28', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (196*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r1 + (196*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.9284766908852592
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = 0.9449111825230679
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp15 = 2.0
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.store(in_out_ptr0 + (r1 + (196*x0)), tmp12, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp28, None)
