
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 256.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp24, xmask)
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr2 + (x0), tmp27, xmask)
