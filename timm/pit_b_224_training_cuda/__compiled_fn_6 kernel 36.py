
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7696
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 962
    x2 = xindex
    y1 = (yindex // 962)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-1) + y0 + (961*x2) + (246016*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x2 + (256*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp13, xmask & ymask)
