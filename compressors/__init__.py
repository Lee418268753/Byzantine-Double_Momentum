from .base_class import Identity, Compressor
from .random_sparsificationlr import RandomSparsifierlr
from .random_sparsificationcnn import RandomSparsifiercnn
from .contractive_compressorlr import TopKCompressorlr
from .contractive_compressorcnn import TopKCompressorcnn
from .contractive_compressorcnn2 import TopKCompressorcnn2
from .contractive_compressorcnn3 import TopKCompressorcnn3

__all__ = ["get_compression", "Identity", "RandomSparsifierlr","RandomSparsifiercnn", "Compressor","TopKCompressorlr","TopKCompressorcnn","TopKCompressorcnn2","TopKCompressorcnn3"]

def get_compression(comp_name):
    if comp_name == 'none':
        return Identity()
    elif comp_name == 'random_sparsificationlr':
        return RandomSparsifierlr()
    elif comp_name == 'random_sparsificationcnn':
        return RandomSparsifiercnn()
    elif comp_name == 'contractive_compressorlr':
        return TopKCompressorlr()
    elif comp_name == 'contractive_compressorcnn':
        return TopKCompressorcnn()
    elif comp_name == 'contractive_compressorcnn2':
        return TopKCompressorcnn2()
    elif comp_name == 'contractive_compressorcnn3':
        return TopKCompressorcnn3()
    else:
        raise ValueError(f"Unknown compression: comp_name")
