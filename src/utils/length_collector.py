import gzip
import io

from biotite.database import rcsb
from biotite.structure import chain_iter, filter_amino_acids, get_residues, get_chains
from biotite.structure.io.pdbx import get_structure, list_assemblies, get_assembly, BinaryCIFFile


def gzip_file_to_binary_stream(file_path):
    """
    Reads a gzip file in binary mode and returns a binary stream (BytesIO).

    Args:
        file_path (str): The path to the gzip file.

    Returns:
        io.BytesIO: A binary stream containing the decompressed data.
                     Returns None if an error occurs during file processing.
    """
    try:
        with gzip.open(file_path, 'rb') as gzipped_file:
            data = gzipped_file.read()
            binary_stream = io.BytesIO(data)
            return binary_stream
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def get_instance_length(pdb_file, asym_id_list):
    bcif = BinaryCIFFile.read(gzip_file_to_binary_stream(pdb_file))
    structure = get_structure(
        bcif,
        use_author_fields=False,
        model=1
    )
    length_list = []
    for atom_ch in chain_iter(structure):
        asym_id = get_chains(atom_ch)[0]
        if asym_id not in asym_id_list:
            continue
        atom_res = atom_ch[filter_amino_acids(atom_ch)]
        if len(atom_res) == 0:
            continue
        res = get_residues(atom_res)
        if res and len(res) > 0:
            length_list.append((asym_id, len(res[0])))
        else:
            length_list.append((asym_id, 0))

    return length_list


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


