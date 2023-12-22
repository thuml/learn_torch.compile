
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mul_native_batch_norm_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49)
    x0 = xindex % 49
    tmp18 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr2 + (x1 + (336*r2) + (43008*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr3 + (x1 + (336*r2) + (43008*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 168, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((784*x1) + (131712*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 336, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((-131712) + (784*x1) + (131712*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp8, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp7, tmp13)
        tmp16 = tmp14 * tmp15
        tmp19 = tmp17 - tmp18
        tmp20 = tmp16 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
