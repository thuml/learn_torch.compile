
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_copy_zeros_like_16', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 513) % 1024
    x1 = xindex % 513
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = x2
    tmp2 = tl.full([1], 768, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = x1
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tmp6 & tmp3
    tmp8 = tl.load(in_ptr1 + ((-197632) + x1 + (257*x2)), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = (tmp8 != 0)
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp10, tmp0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp7, tmp11, tmp12)
    tmp14 = tl.where(tmp6, tmp13, tmp10)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp3, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp16, tmp10)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
    tl.store(out_ptr2 + (x0), tmp17, None)
