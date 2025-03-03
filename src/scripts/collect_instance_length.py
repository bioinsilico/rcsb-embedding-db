import argparse
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.length_collector import get_instance_length

def pdb_file_path(pdb_path, pdb):
    if "_" in pdb:
        return os.path.join(
            pdb_path,
            "csm",
            pdb[0:2],
            pdb[-6:-4],
            pdb[-4:-2],
            f"{pdb}.bcif.gz"
        )
    return os.path.join(pdb_path, "pdb", pdb[1:3], f"{pdb}.bcif.gz")

def process_file(args_tuple):
    filename, asym_id, rcsb_id = args_tuple
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return None
    # Compute the instance length for the file
    length = get_instance_length(filename, asym_id)
    return f"{rcsb_id},{length}\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test length collectors")
    parser.add_argument('--pdb_path', type=str, help="Embeddings folder", required=True)
    parser.add_argument('--instance_list', type=str, help="List of chain file", required=True)
    parser.add_argument('--instance_length_output_file', type=str, help="Instance length output file", required=True)
    args = parser.parse_args()

    pdb_path = args.pdb_path
    instance_list = args.instance_list
    instance_length_output_file = args.instance_length_output_file

    # Prepare the list of file information
    folder_files = [(
        pdb_file_path(pdb_path, rcsb_id.split(".")[0].lower()),
        rcsb_id.split(".")[1].strip(),
        rcsb_id.strip()
    ) for rcsb_id in open(instance_list)]

    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, item): item for item in folder_files}
        with tqdm(total=len(folder_files), desc="Loading instance length", unit="file") as pbar:
            with open(instance_length_output_file, "w") as f:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        f.write(result)
                    pbar.update(1)
