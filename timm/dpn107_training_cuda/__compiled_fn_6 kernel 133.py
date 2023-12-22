
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_132', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6924288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 276
    x2 = (xindex // 865536)
    x3 = xindex % 865536
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (125440 + x3 + (1179136*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (125440 + x3 + (1116416*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (125440 + x3 + (1053696*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (125440 + x3 + (990976*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x3 + (1179136*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (1116416*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (1053696*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (990976*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
