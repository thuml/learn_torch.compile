
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (12544*(r2 // 49)) + (25088*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (256*(r2 // 49)) + (512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (256*(r2 // 49)) + (512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 49.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp18 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, xmask)
