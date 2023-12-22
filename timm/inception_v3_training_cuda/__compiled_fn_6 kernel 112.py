
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (64 + x1 + (288*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (78400 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (78400 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
