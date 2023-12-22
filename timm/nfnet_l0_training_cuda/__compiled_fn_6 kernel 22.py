
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 14
        r2 = (rindex // 14)
        tmp0 = tl.load(in_ptr0 + (r3 + (196*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2))))) >= 0, 0, 7))) + (49*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) >= 0, 0, 7))), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r3 + (196*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r3 + (196*x0)), rmask, eviction_policy='evict_first', other=0.0)
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
        tmp17 = 0.2
        tmp18 = tmp16 * tmp17
        tmp19 = 2.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp26 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = 1.0
    tmp29 = tmp28 - tmp27
    tmp30 = tmp27 * tmp29
    tmp31 = tmp24 * tmp30
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, None)
