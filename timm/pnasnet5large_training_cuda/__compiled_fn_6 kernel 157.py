
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_threshold_backward_156', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    y5 = yindex
    x3 = (xindex // 83)
    x2 = xindex % 83
    tmp0 = tl.load(in_ptr0 + (y0 + (108*x4) + (744012*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x4 + (6889*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = 1 + x3
    tmp5 = tl.full([1, 1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1, 1], 85, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = 1 + x2
    tmp10 = tmp9 >= tmp5
    tmp11 = tmp9 < tmp7
    tmp12 = tmp6 & tmp8
    tmp13 = tmp12 & tmp10
    tmp14 = tmp13 & tmp11
    tmp15 = tl.load(in_ptr1 + (9288 + y0 + (108*x2) + (9180*x3) + (780300*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tmp3 + tmp17
    tmp19 = tl.load(in_ptr2 + (86 + x2 + (85*x3) + (7225*y5)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp0, tmp2, tmp21)
    tmp23 = tmp18 + tmp22
    tmp24 = 2 + x3
    tmp25 = tmp24 >= tmp5
    tmp26 = tl.full([1, 1], 87, tl.int64)
    tmp27 = tmp24 < tmp26
    tmp28 = 2 + x2
    tmp29 = tmp28 >= tmp5
    tmp30 = tmp28 < tmp26
    tmp31 = tmp25 & tmp27
    tmp32 = tmp31 & tmp29
    tmp33 = tmp32 & tmp30
    tmp34 = tl.load(in_ptr3 + (176 + x2 + (87*x3) + (7569*y5)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp33, tmp34, tmp35)
    tmp37 = tl.where(tmp0, tmp2, tmp36)
    tmp38 = tmp23 + tmp37
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4 + (6889*y5)), tmp38, xmask & ymask)
