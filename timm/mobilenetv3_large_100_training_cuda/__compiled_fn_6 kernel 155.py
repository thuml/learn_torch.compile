
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardswish_backward_native_batch_norm_backward_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr3 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp0 / tmp3
        tmp9 = 0.5
        tmp10 = tmp8 + tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.where(tmp4, tmp11, tmp7)
        tmp13 = 0.0
        tmp14 = tl.where(tmp2, tmp13, tmp12)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
