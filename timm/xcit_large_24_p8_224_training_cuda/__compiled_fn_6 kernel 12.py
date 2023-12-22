
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_select_backward_slice_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = (r2 + (126*x1)) % 785
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = tmp6 & tmp2
        tmp8 = tl.load(in_ptr0 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp7 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp11 = 0.0
        tmp12 = tl.where(tmp6, tmp10, tmp11)
        tmp13 = tmp3 + tmp12
        tmp14 = tmp4 < tmp5
        tmp15 = tmp14 & tmp2
        tmp16 = tl.load(in_ptr1 + (x0 + (768*(((r2 + (126*x1)) // 785) % 8))), rmask & tmp15 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tl.where(tmp14, tmp18, tmp11)
        tmp20 = tmp13 + tmp19
        tmp21 = tl.load(in_ptr2 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
        tmp28 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp29 = tl.where(tmp2, tmp20, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp31, xmask)
