import os
import glob

# Clear all cached kernel files to force recomputation
kernel_dir = 'kernels_em_znerem_spambase_final'

if os.path.exists(kernel_dir):
    kernel_files = glob.glob(f'{kernel_dir}/*.npy')
    print(f"Found {len(kernel_files)} kernel files to delete...")
    
    for file in kernel_files:
        os.remove(file)
        print(f"  Deleted: {file}")
    
    print(f"\n✓ Cleared {len(kernel_files)} cached kernel files")
    print("The notebook will now recompute kernels with correct configurations.")
else:
    print(f"Directory {kernel_dir} not found")
