

class EmbeddingLoader:

    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    BATCH_SIZE = 2000

    def __init__(
            self,
            diskann_tmp_folder
    ):
        self.collection = None
        self.diskann_tmp_folder = diskann_tmp_folder

    def save_to_tsv(self, df, prefix):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")
        df[[self.EMBEDDING_FIELD]].to_csv(f"{self.diskann_tmp_folder}/{prefix}.tsv", sep='\t', index=False)
