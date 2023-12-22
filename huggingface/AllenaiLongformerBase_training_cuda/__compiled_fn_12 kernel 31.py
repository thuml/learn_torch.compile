
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 513) % 1024
    x0 = xindex % 513
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = x1
    tmp2 = tl.full([1], 256, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 257, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tmp6 & tmp3
    tmp8 = tl.load(in_ptr1 + (x0 + (257*x1)), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = (tmp8 != 0)
    tmp10 = tl.load(in_out_ptr0 + (x3), tmp7, other=0.0)
    tmp11 = 0.0
    tmp12 = tl.where(tmp9, tmp11, tmp10)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp15 = tl.where(tmp6, tmp14, tmp11)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp3, tmp15, tmp16)
    tmp18 = tl.where(tmp3, tmp17, tmp11)
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
