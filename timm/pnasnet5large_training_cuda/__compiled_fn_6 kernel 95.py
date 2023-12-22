
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_threshold_backward_94', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 432
    y1 = (yindex // 432)
    y5 = yindex
    x3 = (xindex // 42)
    x2 = xindex % 42
    tmp0 = tl.load(in_ptr0 + (y0 + (432*x4) + (762048*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x4 + (1764*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = x3
    tmp5 = tl.full([1, 1], 43, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = x2
    tmp8 = tmp7 < tmp5
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (y0 + (432*x2) + (18576*x3) + (798768*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp3 + tmp12
    tmp14 = tl.load(in_ptr2 + (x2 + (43*x3) + (1849*y5)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tl.where(tmp0, tmp2, tmp16)
    tmp18 = tmp13 + tmp17
    tmp19 = 1 + x3
    tmp20 = tl.full([1, 1], 0, tl.int64)
    tmp21 = tmp19 >= tmp20
    tmp22 = tl.full([1, 1], 45, tl.int64)
    tmp23 = tmp19 < tmp22
    tmp24 = 1 + x2
    tmp25 = tmp24 >= tmp20
    tmp26 = tmp24 < tmp22
    tmp27 = tmp21 & tmp23
    tmp28 = tmp27 & tmp25
    tmp29 = tmp28 & tmp26
    tmp30 = tl.load(in_ptr3 + (46 + x2 + (45*x3) + (2025*y5)), tmp29 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tl.where(tmp0, tmp2, tmp32)
    tmp34 = tmp18 + tmp33
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4 + (1764*y5)), tmp34, xmask & ymask)
