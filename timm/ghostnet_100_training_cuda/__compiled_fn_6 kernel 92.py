
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3120
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (240*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (47040 + (196*x1) + (94080*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (240 + x1 + (480*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.load(in_ptr3 + (240 + x1 + (480*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = 0.0
        tmp12 = tl.where(tmp3, tmp11, tmp10)
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
