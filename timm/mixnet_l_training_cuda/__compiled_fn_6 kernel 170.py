
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_169', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp30 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (4816896*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp29 = tl.load(in_ptr4 + (x0 + (192*r2) + (4816896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 64, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x0) + (802816*(r2 // 12544)) + (1605632*x1) + (r2 % 12544)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 128, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tmp9 & tmp11
        tmp13 = tl.load(in_ptr2 + ((-802816) + (12544*x0) + (802816*(r2 // 12544)) + (1605632*x1) + (r2 % 12544)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp12, tmp13, tmp14)
        tmp16 = tmp1 >= tmp10
        tmp17 = tl.full([1, 1], 192, tl.int64)
        tmp18 = tmp1 < tmp17
        tmp19 = tl.load(in_ptr3 + ((-1605632) + (12544*x0) + (802816*(r2 // 12544)) + (1605632*x1) + (r2 % 12544)), rmask & tmp16 & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp16, tmp19, tmp20)
        tmp22 = tl.where(tmp12, tmp15, tmp21)
        tmp23 = tl.where(tmp5, tmp8, tmp22)
        tmp24 = 0.0
        tmp25 = tl.where(tmp0, tmp24, tmp23)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp31 = tmp29 - tmp30
        tmp32 = tmp25 * tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp34, xmask)
