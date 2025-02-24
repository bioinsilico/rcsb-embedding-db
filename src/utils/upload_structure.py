import torch
from biotite.structure import chain_iter, get_residues, filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, BinaryCIFFile
from esm.models.esm3 import ESM3, ESM3_OPEN_SMALL, ESM3InferenceClient
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.structure.protein_chain import ProteinChain

from utils.load_aggregator import load_aggregator


def get_structure_from_stream(file_stream, format="PDB", chain_id=None):
    if format == "pdb":
        structure = PDBFile.read(file_stream).get_structure(
            model=1
        )
    elif format == "mmcif":
        cif_file = CIFFile.read(file_stream)
        structure = get_structure(
            cif_file,
            model=1,
            use_author_fields=False
        )
    elif format == "binarycif":
        cif_file = BinaryCIFFile.read(file_stream)
        structure = get_structure(
            cif_file,
            model=1,
            use_author_fields=False
        )

    if chain_id:
        structure = structure[structure.chain_id == chain_id]
    return structure


def get_embedding_method(model_path):
    aggregator = load_aggregator(
        model_path
    )
    aggregator.eval()
    esm3_model: ESM3InferenceClient = ESM3.from_pretrained(ESM3_OPEN_SMALL)

    def __compute_embeddings(structure):
        embedding_ch = []
        for atom_ch in chain_iter(structure):
            atom_res = atom_ch[filter_amino_acids(atom_ch)]
            if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
                continue
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            protein = ESMProtein.from_protein_chain(protein_chain)
            protein_tensor = esm3_model.encode(protein)
            embedding_ch.append( esm3_model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding)
        embedding_ch = torch.cat(
            embedding_ch,
            dim=0
        )
        with torch.no_grad():
            return aggregator.embedding(aggregator.transformer(embedding_ch).sum(dim=0)).numpy()

    return __compute_embeddings
