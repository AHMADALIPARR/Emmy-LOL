extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x0, const float* x1, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 500) {
        int a0 = _gid;
        float in0__u0 = x0[a0 * 2];
        float in1__u0 = x1[a0 * 2];
        float v0__u0 = in0__u0 + in1__u0;
        float in0__u1 = x0[a0 * 2 + 1];
        float in1__u1 = x1[a0 * 2 + 1];
        float v0__u1 = in0__u1 + in1__u1;
        float2 _vs_v0__u0_59850f2f = make_float2(v0__u0, v0__u1);
        *reinterpret_cast<float2*>(&add[a0 * 2]) = _vs_v0__u0_59850f2f;
    }
}
