
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_native_layer_norm_backward_167', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (64 + x0 + (64*((r2 + (128*x1)) % 3136)) + (200768*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp4 = 1 + ((r2 + (128*x1)) % 3136)
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = tl.load(in_ptr0 + (64 + x0 + (64*((r2 + (128*x1)) % 3136)) + (200768*((r2 + (128*x1)) // 3136))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + ((3136*x0) + (200704*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp4 < tmp5
        tmp15 = tl.load(in_ptr0 + (x0 + (200768*((r2 + (128*x1)) // 3136))), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp14, tmp15, tmp16)
        tmp18 = tl.where(tmp14, tmp17, tmp12)
        tmp19 = tmp13 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
