
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (1024*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (1024*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 49.0
        tmp16 = tmp14 / tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 1024.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tmp27 = 0.8999999985098839
        tmp28 = tmp26 / tmp27
        tmp29 = tmp25 * tmp28
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp25, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (1024*x3)), tmp29, rmask & xmask)
