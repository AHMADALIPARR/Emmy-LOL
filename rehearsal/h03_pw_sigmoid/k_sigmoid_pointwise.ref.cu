extern "C" __global__
__launch_bounds__(256) void k_sigmoid_pointwise(const float* x, float* sigmoid) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 683) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 3];
        float v0__u0 = 1.0f / (1.0f + expf(0.0f - in0__u0));
        float in0__u1 = x[a0 * 3 + 1];
        float v0__u1 = 1.0f / (1.0f + expf(0.0f - in0__u1));
        float in0__u2 = x[a0 * 3 + 2];
        float v0__u2 = 1.0f / (1.0f + expf(0.0f - in0__u2));
        sigmoid[a0 * 3] = v0__u0;
        sigmoid[a0 * 3 + 1] = v0__u1;
        sigmoid[a0 * 3 + 2] = v0__u2;
    }
}
