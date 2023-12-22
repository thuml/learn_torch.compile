
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_neg_sum_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    x0 = xindex % 7
    x3 = (xindex // 5376)
    x6 = (xindex // 7) % 768
    x1 = (xindex // 7) % 48
    x2 = (xindex // 336) % 16
    tmp3 = tl.load(in_ptr2 + (x2 + (16*x1) + (768*x3)), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r4 + (112*x5)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr1 + (x6 + (2304*r4) + (258048*x0) + (1806336*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -tmp0
        tmp4 = 1e-12
        tmp5 = triton_helpers.maximum(tmp3, tmp4)
        tmp6 = tmp2 / tmp5
        tmp7 = tmp6 / tmp5
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp10, None)
