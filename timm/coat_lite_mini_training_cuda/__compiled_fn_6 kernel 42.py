
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 4) % 49
    x2 = (xindex // 196)
    x4 = xindex % 196
    x0 = xindex % 4
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp16 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = 1 + x1
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (512 + r3 + (128*x4) + (25600*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (49*r3) + (6272*x0) + (25088*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tmp0 < tmp1
        tmp11 = tl.load(in_ptr0 + (r3 + (128*x0) + (25600*x2)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tl.where(tmp10, tmp13, tmp8)
        tmp15 = tmp9 + tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp19, xmask)
