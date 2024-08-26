from keras.src.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from transformer.mlp import MLP
from transformer.patch_encoder import PatchEncoder
from transformer.patch_extract_layer import PatchExtractor
from transformer.transformer_encoder import TransformerEncoder


def create_vit_model(num_classes, num_patches=196, projection_dim=768, input_shape=(224, 224, 1)):
    inputs = Input(shape=input_shape)
    # Patch extractor
    patches = PatchExtractor()(inputs)
    # Patch encoder
    patches_embed = PatchEncoder(num_patches, projection_dim)(patches)
    # Transformer encoder
    representation = TransformerEncoder(projection_dim)(patches_embed)
    representation = GlobalAveragePooling1D()(representation)
    # MLP to classify outputs
    logits = MLP(projection_dim, num_classes, 0.5)(representation)
    # Create model
    model = Model(inputs=inputs, outputs=logits)
    return model
