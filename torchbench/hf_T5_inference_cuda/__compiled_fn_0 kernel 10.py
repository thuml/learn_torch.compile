
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 32128
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp0 + 32128
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert((0 <= tmp16) & (tmp16 < 32128), "index out of bounds: 0 <= tmp16 < 32128")
        tmp17 = tl.load(in_ptr1 + (r1 + (512*tmp16)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp21 = tmp19 + tmp20
        tmp22 = 512.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-06
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp13 * tmp27
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp28, rmask)
