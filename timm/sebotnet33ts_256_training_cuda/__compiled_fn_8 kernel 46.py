
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex % 256
    tmp0 = x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*y1) + (1024*y0) + (16384*(((y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((64*y0) + (1024*y1) + (16384*(((y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((64*y0) + (1024*y1) + (16384*(((y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 512, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + ((-65536) + y4 + (256*x3) + (65536*y2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = tmp0 >= tmp13
    tmp20 = tl.full([1, 1], 768, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((64*y0) + (1024*y1) + (16384*((((-131072) + y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp18, tmp24)
    tmp26 = tl.where(tmp4, tmp11, tmp25)
    tl.store(out_ptr0 + (y4 + (256*x3) + (196608*y2)), tmp26, xmask)
