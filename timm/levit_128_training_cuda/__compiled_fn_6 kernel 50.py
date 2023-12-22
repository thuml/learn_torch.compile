
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r2 % 49) // 7) % 2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.broadcast_to(((r2 % 49) % 7) % 2, [XBLOCK, RBLOCK])
        tmp4 = tmp3 == tmp1
        tmp5 = tmp4 & tmp2
        tmp6 = tl.load(in_ptr0 + (x0 + (256*(((r2 % 49) % 7) // 2)) + (1024*((r2 % 49) // 14)) + (4096*(r2 // 49)) + (8192*x1)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = 0.0
        tmp10 = tl.where(tmp4, tmp8, tmp9)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.where(tmp2, tmp12, tmp9)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp17 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp26, xmask)
