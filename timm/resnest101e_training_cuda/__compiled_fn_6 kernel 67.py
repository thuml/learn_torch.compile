
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (4096*(x0 % 128)) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((128*r2) + (x0 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 4096.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp19, xmask)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr2 + (x0), tmp22, xmask)
