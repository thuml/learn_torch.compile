
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_leaky_relu_backward_native_batch_norm_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (524288*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.01
        tmp5 = tmp3 * tmp4
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)