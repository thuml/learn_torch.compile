
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr2 + (x0 + (2048*(r2 // 64)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + ((64*x0) + (131072*(r2 // 64)) + (262144*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr6 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = 64.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.where(tmp3, tmp1, tmp6)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp10 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp26, None)
