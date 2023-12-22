
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*(tl.math.min(tl.math.max(0, (((r2 + (128*x1)) % 28) // 2)), (-1) + (tl.math.min(14, 1 + (((r2 + (128*x1)) % 28) // 2)))))) + (192*(tl.where((tl.math.min(tl.math.max(0, (((r2 + (128*x1)) % 28) // 2)), (-1) + (tl.math.min(14, 1 + (((r2 + (128*x1)) % 28) // 2))))) >= 0, 0, 14))) + (2688*(tl.math.min(tl.math.max(0, ((((r2 + (128*x1)) // 28) % 28) // 2)), (-1) + (tl.math.min(14, 1 + ((((r2 + (128*x1)) // 28) % 28) // 2)))))) + (2688*(tl.where((tl.math.min(tl.math.max(0, ((((r2 + (128*x1)) // 28) % 28) // 2)), (-1) + (tl.math.min(14, 1 + ((((r2 + (128*x1)) // 28) % 28) // 2))))) >= 0, 0, 14))) + (37632*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr1 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0 / 4
        tmp2 = tl.math.max(0, ((((r2 + (128*x1)) // 28) % 28) // 2))
        tmp3 = tl.math.min(14, 1 + ((((r2 + (128*x1)) // 28) % 28) // 2))
        tmp4 = tmp2 < tmp3
        tmp5 = tl.math.max(0, (((r2 + (128*x1)) % 28) // 2))
        tmp6 = tl.math.min(14, 1 + (((r2 + (128*x1)) % 28) // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tmp4 & tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp8, tmp1, tmp9)
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp18 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
