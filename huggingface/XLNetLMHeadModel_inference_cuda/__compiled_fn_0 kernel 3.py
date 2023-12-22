
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_index_select_mul_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (512 + r2 + (1023*x0) + (524288*x1) + (524288*((r2 + (1023*x0)) // 523776))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = 0.125
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr1 + (512 + r2 + (1023*x0) + (524288*x1) + (524288*((r2 + (1023*x0)) // 523776))), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = 0.0
        tmp14 = tmp12 + tmp13
        tmp15 = 0.125
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = tl.exp(tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr1 + (512 + r2 + (1023*x0) + (524288*x1) + (524288*((r2 + (1023*x0)) // 523776))), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = 0.0
        tmp26 = tmp24 + tmp25
        tmp27 = 0.125
        tmp28 = tmp26 * tmp27
        tmp29 = tmp28 - tmp8
        tmp30 = tl.exp(tmp29)
        tmp31 = tmp30 / tmp20
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask)
