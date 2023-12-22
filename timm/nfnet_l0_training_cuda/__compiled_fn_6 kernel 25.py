
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
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
    r3 = rindex
    x0 = xindex
    r1 = rindex % 14
    r2 = (rindex // 14)
    tmp0 = tl.load(in_out_ptr0 + (r3 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2))))) >= 0, 0, 7))) + (49*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) >= 0, 0, 7))), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr2 + (r3 + (196*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r3 + (196*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr4 + (r3 + (196*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (r2 // 2))
    tmp4 = tl.math.min(7, 1 + (r2 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (r1 // 2))
    tmp7 = tl.math.min(7, 1 + (r1 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.8980265101338745
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = 0.9128709291752768
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = 0.2
    tmp24 = tmp22 * tmp23
    tmp25 = 2.0
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = 1.0
    tmp36 = tmp35 - tmp34
    tmp37 = tmp34 * tmp36
    tmp38 = tmp32 * tmp37
    tl.store(in_out_ptr0 + (r3 + (196*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp38, None)
