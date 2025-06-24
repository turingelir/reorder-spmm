from ssgetpy import search, fetch

def get_matrices(path='/home/cak/fatih/dev/spring2025/parallel_computing_for_gpus/project/spECK/data'):
    # Filter: symmetric matrices with dimensions between 1k and 1M
    results = search(rowbounds=(32_000, 48_000), limit=500) # , kind='symmetric'

    print(f"Found {len(results)} matrices with 1k to 1M non-zeros.")

    # Estimate size (rough formula):
    # Each nonzero entry is written as "row col value\n", so 20–40 bytes per entry
    # Header takes a few hundred bytes
    def estimate_mtx_size_bytes(mtx):
        approx_bytes_per_nnz = 28  # conservative plain-text estimation
        overhead = 500             # header + coordinate count line
        return mtx.nnz * approx_bytes_per_nnz + overhead

    # Sum up total estimated size
    total_bytes = sum(estimate_mtx_size_bytes(m) for m in results)

    # Convert to GB
    total_gb = total_bytes / 1024**3

    print(f"Estimated total download size: {total_gb:.2f} GB")

    # Download each matrix
    for mtx in results:
        print(f"Downloading {mtx.name} (ID: {mtx.id})...")
        if mtx.id < 560:
            continue
        fetch(mtx.id, format='MM', location=path)
        print(f"Downloaded {mtx.name} to {path}/{mtx.name}.mtx")
        print("...\n")
    print("-----All matrices downloaded successfully.-----")

def get(path='/home/cak/fatih/dev/spring2025/parallel_computing_for_gpus/project/spECK/data'):
    # Download the first 1000 matrices that match the criteria
    results = search(rowbounds=(48_000, 64_000), limit=1000)

    for mtx in results:
        print(f"Downloading {mtx.name} (ID: {mtx.matid})...")
        fetch(mtx.matid, format='MM', location=path)
        print(f"Downloaded {mtx.name} to {path}/{mtx.name}.mtx")
        print("...\n")
    print("-----All matrices downloaded successfully.-----")

if __name__ == "__main__":
    get_matrices(path='/home/cak/fatih/dev/spring2025/parallel_computing_for_gpus/project/spECK/data')

    