
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 58
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (58*r3)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp23 = tl.load(in_ptr4 + (x0 + (58*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 1 + (2*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 58, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + (r1 + (784*((1 + (2*x0)) // 58)) + (1568*((1 + (2*x0)) % 58)) + (90944*r2)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*((1 + (2*x0)) // 58)) + (1568*((1 + (2*x0)) % 58)) + (90944*r2)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
        tmp10 = tl.where(tmp5, tmp8, tmp9)
        tmp11 = tmp1 >= tmp4
        tmp12 = tl.full([1, 1], 116, tl.int64)
        tmp13 = tmp1 < tmp12
        tmp14 = tl.load(in_ptr3 + ((-44688) + r1 + (1568*x0) + (45472*r2)), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp11, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp10, tmp16)
        tmp18 = 0.0
        tmp19 = tl.where(tmp0, tmp18, tmp17)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
        tmp25 = tmp23 - tmp24
        tmp26 = tmp19 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp30 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp34, xmask)
