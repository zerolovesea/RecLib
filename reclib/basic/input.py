import torch.nn as nn

from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature

def init_embedding_layers(dense_features, sparse_features, sequence_features):
    sparse_columns = [f for f in sparse_features if isinstance(f, SparseFeature)]
    sequence_columns = [f for f in sequence_features if isinstance(f, SequenceFeature)]
    all_emb_features = sparse_columns + sequence_columns

    embedding_dict = nn.ModuleDict()

    for feat in all_emb_features:
        emb_name = feat.embedding_name
        if emb_name in embedding_dict:
            continue
        
        emb = nn.Embedding(
            num_embeddings=feat.vocab_size,
            embedding_dim=feat.embedding_dim,
            padding_idx=feat.padding_idx,
        )

        if feat.init_type == "uniform":
            nn.init.uniform_(emb.weight)
        elif feat.init_type == "normal":
            nn.init.normal_(emb.weight)
        elif feat.init_type == "xavier_uniform":
            nn.init.xavier_uniform_(emb.weight)
        elif feat.init_type == "xavier_normal":
            nn.init.xavier_normal_(emb.weight)
        else:
            raise ValueError(f"Unknown initializer: {feat.init_type}")

        emb.weight.requires_grad = feat.trainable
        embedding_dict[feat.name] = emb

    return embedding_dict
