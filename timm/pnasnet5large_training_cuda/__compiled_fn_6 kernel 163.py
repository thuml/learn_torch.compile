
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_162', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 46548
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 108)
    x0 = xindex % 108
    _tmp45 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 55112, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((6889*x0) + (744012*(((r2 + (128*x1)) // 6889) % 8)) + ((r2 + (128*x1)) % 6889)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 1 + (((r2 + (128*x1)) // 83) % 83)
        tmp5 = tl.full([1, 1], 0, tl.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = tl.full([1, 1], 85, tl.int64)
        tmp8 = tmp4 < tmp7
        tmp9 = 1 + ((r2 + (128*x1)) % 83)
        tmp10 = tmp9 >= tmp5
        tmp11 = tmp9 < tmp7
        tmp12 = tmp6 & tmp8
        tmp13 = tmp12 & tmp10
        tmp14 = tmp13 & tmp11
        tmp15 = tmp14 & tmp2
        tmp16 = tl.load(in_ptr1 + (9288 + x0 + (108*((r2 + (128*x1)) % 83)) + (9180*(((r2 + (128*x1)) // 83) % 83)) + (780300*(((r2 + (128*x1)) // 6889) % 8))), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tmp3 + tmp18
        tmp20 = tl.load(in_ptr2 + (x0 + (108*((r2 + (128*x1)) % 55112))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp21 = 3 + (((r2 + (128*x1)) // 83) % 83)
        tmp22 = tmp21 >= tmp5
        tmp23 = tl.full([1, 1], 89, tl.int64)
        tmp24 = tmp21 < tmp23
        tmp25 = 3 + ((r2 + (128*x1)) % 83)
        tmp26 = tmp25 >= tmp5
        tmp27 = tmp25 < tmp23
        tmp28 = tmp22 & tmp24
        tmp29 = tmp28 & tmp26
        tmp30 = tmp29 & tmp27
        tmp31 = tmp30 & tmp2
        tmp32 = tl.load(in_ptr3 + (270 + (89*(((r2 + (128*x1)) // 83) % 83)) + (7921*x0) + (855468*(((r2 + (128*x1)) // 6889) % 8)) + ((r2 + (128*x1)) % 83)), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp31, tmp32, tmp33)
        tmp35 = 0.0
        tmp36 = tl.where(tmp20, tmp35, tmp34)
        tmp37 = tmp19 + tmp36
        tmp38 = tl.load(in_ptr4 + (x0 + (108*((r2 + (128*x1)) % 55112))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tmp38 - tmp39
        tmp41 = tmp37 * tmp40
        tmp42 = tl.full(tmp41.shape, 0, tmp41.dtype)
        tmp43 = tl.where(tmp2, tmp41, tmp42)
        tmp44 = tl.broadcast_to(tmp43, [XBLOCK, RBLOCK])
        tmp46 = _tmp45 + tmp44
        _tmp45 = tl.where(rmask & xmask, tmp46, _tmp45)
    tmp45 = tl.sum(_tmp45, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp45, xmask)
