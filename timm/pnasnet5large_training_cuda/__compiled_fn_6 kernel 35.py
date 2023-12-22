
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6912
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 864
    x1 = (xindex // 864)
    x3 = xindex
    tmp18 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (418176 + r2 + (121*x0) + (522720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (864*r2) + (104544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (864*r2) + (104544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_out_ptr0 + (r2 + (121*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r2 + (121*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (864*r2) + (104544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr5 + (r2 + (121*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr6 + (x0 + (864*r2) + (104544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tmp2 + tmp7
        tmp10 = tl.where(tmp5, tmp4, tmp9)
        tmp11 = tmp8 + tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tl.where(tmp5, tmp4, tmp14)
        tmp16 = tmp13 + tmp15
        tmp19 = tmp17 - tmp18
        tmp20 = tmp16 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tl.store(in_out_ptr0 + (r2 + (121*x3)), tmp16, rmask & xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
