import os
from netCDF4 import Dataset
import sys

# === CONFIGURATION ===
# Using the path from your error logs
TARGET_DIR = "/Users/andrewxu/Documents/Python/axu-trading/data/era5_raw"

def check_files(directory):
    print(f"🔍 Scanning directory: {directory}")
    print("-" * 50)
    
    corrupted_files = []
    total_files = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".nc"):
                total_files += 1
                file_path = os.path.join(root, file)
                
                try:
                    # Attempt to open the file in read-mode
                    # If the HDF5 header is corrupt, this lines throws OSError immediately
                    with Dataset(file_path, mode='r') as ds:
                        # Optional: Check if variables actually exist (deeper check)
                        if not ds.variables:
                            print(f"⚠️  [EMPTY] {file} (Opens but has no variables)")
                        else:
                            # print(f"✅ [OK] {file}") # Uncomment if you want to see successes
                            pass
                            
                except OSError as e:
                    print(f"❌ [CORRUPT] {file}")
                    print(f"    └─ Error: {e}")
                    corrupted_files.append(file_path)
                except Exception as e:
                    print(f"❌ [ERROR] {file} - Unexpected error: {e}")
                    corrupted_files.append(file_path)

    print("-" * 50)
    print(f"Scan complete.")
    print(f"Total files checked: {total_files}")
    print(f"Corrupted files found: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\n👇 List of Corrupted Files (Copy these to delete):")
        for f in corrupted_files:
            print(f)
            
        print("\n💡 Tip: To delete all of them at once, you can run:")
        print("rm " + " ".join(f'"{f}"' for f in corrupted_files))

if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory not found: {TARGET_DIR}")
    else:
        check_files(TARGET_DIR)