
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r1 + (144*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.9128709291752768
        tmp3 = tmp1 * tmp2
        tmp4 = 1.7015043497085571
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = 0.2
        tmp10 = tmp8 * tmp9
        tmp13 = tmp10 * tmp12
        tmp14 = 2.0
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = 1.0
    tmp24 = tmp23 - tmp22
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp26, None)
