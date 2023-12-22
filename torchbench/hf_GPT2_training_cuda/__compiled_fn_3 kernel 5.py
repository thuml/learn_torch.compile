
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 8.0
        tmp3 = tmp1 / tmp2
        tmp4 = -3.4028234663852886e+38
        tmp5 = tl.where(tmp0, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp10 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = -3.4028234663852886e+38
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tmp14 - tmp7
        tmp16 = tl.exp(tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp21 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp22 = 8.0
        tmp23 = tmp21 / tmp22
        tmp24 = -3.4028234663852886e+38
        tmp25 = tl.where(tmp20, tmp23, tmp24)
        tmp26 = tmp25 - tmp7
        tmp27 = tl.exp(tmp26)
        tmp28 = tmp27 / tmp18
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp28, rmask)
        tl.store(out_ptr3 + (r2 + (512*x3)), tmp28, rmask)
