
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x1) + (188160*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (240*(((r2 + (126*x0)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x1 + (240*(((r2 + (126*x0)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 784.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (240*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
