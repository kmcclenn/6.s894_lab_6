def main():
    sizes = [
        [3072, 3072, 3072],
        [2048, 3072, 3072],
        [1024, 3072, 3072],
        [512, 3072, 3072],
        [256, 3072, 3072],
        [128, 3072, 3072],
        [64, 3072, 3072],
        [32, 3072, 3072],
        [16, 3072, 3072],
    ]

    fma_tflops_s = [26.73, 26.73, 26.73, 26.73, 26.73, 21.27, 11.06, 5.64, 2.85]

    max_tput_tflops = 53.45
    mem_bw_mb = 360000  # mb/s

    for (i, j, k), fma_tflop in zip(sizes, fma_tflops_s):
        print(f"Size: i = {i}, j = {j}, k = {k}")
        total_tflops = i * j * k * 2 / (10**12)  # for multiply + add

        mb_needed = (i * k + k * j + i * j) * 4 / (10**6)
        total_time_mem_ms = mb_needed / mem_bw_mb * 1000
        total_time_tensor_ms = total_tflops / max_tput_tflops * 1000
        bottleneck = max(total_time_tensor_ms, total_time_mem_ms)
        tflop_s = 1000 * total_tflops / bottleneck

        print(
            f"1. Time (ms) if tensor core limited: {round(total_time_tensor_ms, 5)} ms."
        )
        print(
            f"2. Time (ms) if memory limited: {round(total_time_mem_ms, 5)} ms. So, lower bound is {round(bottleneck, 5)} ms."
        )
        print(f"3. Max TFLOP/s: {round((tflop_s), 5)} TFLOP/s")
        if total_time_tensor_ms > total_time_mem_ms:
            print(f"Thus, workload (compute) bound.")
            if tflop_s > fma_tflop:
                print(f"4. Larger max throughput than with FMAs")
            else:
                print(f"4. Smaller max throughput than with FMAs")
            if i >= 256:
                print("Both compute bound.")
            else:
                print("Tensor core compute bound, FMA is memory bound")
        else:
            print(f"Thus, bandwidth (memory) bound.")
            print(f"4. Same max throughput as FMA")
            if i < 256:
                print("Both memory bound.")
            else:
                print("Tensor core memory bound, FMA is compute bound")

        print("\n")
if __name__ == "__main__":
    main()
