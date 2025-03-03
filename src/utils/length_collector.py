import argparse
import os

from biotite.database import rcsb
from biotite.structure import chain_iter, filter_amino_acids, get_residues, get_chains
from biotite.structure.io.pdbx import get_structure, list_assemblies, get_assembly, BinaryCIFFile
from tqdm import tqdm


def get_instance_length(rcsb_id):
    asym_id = rcsb_id.split(".")[1] if "." in rcsb_id else "A"
    pdb = rcsb_id.split(".")[0]
    rcsb_fetch = rcsb.fetch(pdb, "bcif")
    bcif = BinaryCIFFile.read(rcsb_fetch)
    structure = get_structure(
        bcif,
        use_author_fields=False,
        model=1
    )
    structure = structure[structure.chain_id == asym_id]
    for atom_ch in chain_iter(structure):
        atom_res = atom_ch[filter_amino_acids(atom_ch)]
        if len(atom_res) == 0:
            continue
        res = get_residues(atom_res)
        if res and len(res) > 0:
            return len(res[0])
    return 0


def get_assembly_length(rcsb_id):
    pdb = rcsb_id.split("-")[0]
    assembly_id = rcsb_id.split("-")[1]
    rcsb_fetch = rcsb.fetch(pdb, "bcif")
    bcif = BinaryCIFFile.read(rcsb_fetch)
    for _assembly_id in list_assemblies(bcif):
        if assembly_id != _assembly_id:
            continue
        atom_array = get_assembly(
            bcif,
            model=1,
            assembly_id=assembly_id,
            use_author_fields=False
        )
        assembly_len = 0
        for atom_ch in chain_iter(atom_array):
            ch = get_chains(atom_ch)[0]
            atom_res = atom_ch[filter_amino_acids(atom_ch)]
            if len(atom_res) == 0:
                continue
            res = get_residues(atom_res)
            if res and len(res) > 0:
                assembly_len += len(res[0])
        return assembly_len
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test length collectors")
    parser.add_argument('--folder_path', type=str, help="Embeddings folder", required=True)
    args = parser.parse_args()
    folder_path = args.folder_path

    folder_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    with tqdm(total=len(folder_files), desc="Loading embeddings", unit="file") as pbar:
        for filename in folder_files:
            rcsb_id, _ = os.path.splitext(filename)
            print(rcsb_id, get_instance_length(rcsb_id))
            pbar.update(1)
