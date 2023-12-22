
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_eq_fill_lift_fresh_masked_fill_new_zeros_select_scatter_slice_scatter_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tl.load(in_ptr0 + ((-1) + x0), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tl.full([1], 2, tl.int64)
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp12 = tl.full([1], -100, tl.int64)
    tmp13 = tmp11 == tmp12
    tmp14 = tl.where(tmp13, tmp3, tmp11)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
