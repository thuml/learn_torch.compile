
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-50176) + x3 + (175616*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (150528 + x3 + (363776*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr2 + (x3 + (200704*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (363776*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
