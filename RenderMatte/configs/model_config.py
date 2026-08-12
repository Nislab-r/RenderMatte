from typing_extensions import Literal, TypeAlias

# from models.sd_text_encoder import SDTextEncoder
# from models.sd_unet_abc import SDUNet
# from models.sd_vae_encoder import SDVAEEncoder
# from models.sd_vae_decoder import SDVAEDecoder

# from models.sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
# from models.sdxl_unet import SDXLUNet
# from models.sdxl_vae_decoder import SDXLVAEDecoder
# from models.sdxl_vae_encoder import SDXLVAEEncoder

# from models.sd3_text_encoder import SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3
from models.flux_text_encoder import SD3TextEncoder1
# from models.sd3_dit_abc import SD3DiT
# from models.sd3_vae_decoder import SD3VAEDecoder
# from models.sd3_vae_encoder import SD3VAEEncoder

# from models.sd_controlnet import SDControlNet
# from models.sdxl_controlnet import SDXLControlNetUnion

# from models.sd_motion import SDMotionModel
# from models.sdxl_motion import SDXLMotionModel

# from models.svd_image_encoder import SVDImageEncoder
# from models.svd_unet import SVDUNet
# from models.svd_vae_decoder import SVDVAEDecoder
# from models.svd_vae_encoder import SVDVAEEncoder

# from models.sd_ipadapter import SDIpAdapter, IpAdapterCLIPImageEmbedder
# from models.sdxl_ipadapter import SDXLIpAdapter, IpAdapterXLCLIPImageEmbedder

# from models.hunyuan_dit_text_encoder import HunyuanDiTCLIPTextEncoder, HunyuanDiTT5TextEncoder
# from models.hunyuan_dit import HunyuanDiT

from models.flux_dit import FluxDiT
from models.flux_text_encoder import FluxTextEncoder2
from models.flux_vae import FluxVAEEncoder, FluxVAEDecoder
# from models.flux_controlnet import FluxControlNet
# from models.flux_ipadapter import FluxIpAdapter
# from models.flux_infiniteyou import InfiniteYouImageProjector

# from models.cog_vae import CogVAEEncoder, CogVAEDecoder
# from models.cog_dit import CogDiT

# from models.abc import OmniGenTransformer

# from models.hunyuan_video_vae_decoder import HunyuanVideoVAEDecoder
# from models.hunyuan_video_vae_encoder import HunyuanVideoVAEEncoder

# # from extensions.RIFE import IFNet
# # from extensions.ESRGAN import RRDBNet

# from models.hunyuan_video_dit import HunyuanVideoDiT

# from models.stepvideo_vae import StepVideoVAE
# from models.stepvideo_dit import StepVideoModel

# from models.wan_video_dit import WanModel
# from models.wan_video_dit_s2v import WanS2VModel
# from models.wan_video_text_encoder import WanTextEncoder
# from models.wan_video_image_encoder import WanImageEncoder
# from models.wan_video_vae import WanVideoVAE, WanVideoVAE38
# from models.wan_video_motion_controller import WanMotionControllerModel
# from models.wan_video_vace import VaceWanModel
# from models.wav2vec import WanS2VAudioEncoder

# from models.step1x_connector import Qwen2Connector

# from models.flux_value_control import SingleValueEncoder

# from lora.flux_lora import FluxLoraPatcher
# from models.flux_lora_encoder import FluxLoRAEncoder

# from models.nexus_gen_projector import NexusGenAdapter, NexusGenImageEmbeddingMerger
# from models.nexus_gen import NexusGenAutoregressiveModel

# from models.qwen_image_dit import QwenImageDiT
# from models.qwen_image_text_encoder import QwenImageTextEncoder
# from models.qwen_image_vae import QwenImageVAE
# from models.qwen_image_controlnet import QwenImageBlockWiseControlNet

model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    # (None, "091b0e30e77c76626b3ba62acdf95343", ["sd_controlnet"], [SDControlNet], "civitai"),
    # (None, "4a6c8306a27d916dea81263c8c88f450", ["hunyuan_dit_clip_text_encoder"], [HunyuanDiTCLIPTextEncoder], "civitai"),
    # (None, "f4aec400fe394297961218c768004521", ["hunyuan_dit"], [HunyuanDiT], "civitai"),
    # (None, "9e6e58043a5a2e332803ed42f6ee7181", ["hunyuan_dit_t5_text_encoder"], [HunyuanDiTT5TextEncoder], "civitai"),
    # (None, "13115dd45a6e1c39860f91ab073b8a78", ["sdxl_vae_encoder", "sdxl_vae_decoder"], [SDXLVAEEncoder, SDXLVAEDecoder], "diffusers"),
    # (None, "d78aa6797382a6d455362358a3295ea9", ["sd_ipadapter_clip_image_encoder"], [IpAdapterCLIPImageEmbedder], "diffusers"),
    # (None, "e291636cc15e803186b47404262ef812", ["sd_ipadapter"], [SDIpAdapter], "civitai"),
    # (None, "399c81f2f8de8d1843d0127a00f3c224", ["sdxl_ipadapter_clip_image_encoder"], [IpAdapterXLCLIPImageEmbedder], "diffusers"),
    # (None, "a64eac9aa0db4b9602213bc0131281c7", ["sdxl_ipadapter"], [SDXLIpAdapter], "civitai"),
    # (None, "52817e4fdd89df154f02749ca6f692ac", ["sdxl_unet"], [SDXLUNet], "diffusers"),
    # (None, "03343c606f16d834d6411d0902b53636", ["sd_text_encoder", "sd_unet", "sd_vae_decoder", "sd_vae_encoder"], [SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder], "civitai"),
    # (None, "d4ba77a7ece070679b4a987f58f201e9", ["sd_text_encoder"], [SDTextEncoder], "civitai"),
    # (None, "d0c89e55c5a57cf3981def0cb1c9e65a", ["sd_vae_decoder", "sd_vae_encoder"], [SDVAEDecoder, SDVAEEncoder], "civitai"),
    # (None, "3926bf373b39a67eeafd7901478a47a7", ["sd_unet"], [SDUNet], "civitai"),
    # (None, "1e0c39ec176b9007c05f76d52b554a4d", ["sd3_text_encoder_1", "sd3_text_encoder_2", "sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3TextEncoder1, SD3TextEncoder2, SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    (None, "1e0c39ec176b9007c05f76d52b554a4d", ["sd3_text_encoder_1"], [SD3TextEncoder1], "civitai"),
    # (None, "d9e0290829ba8d98e28e1a2b1407db4a", ["sd3_text_encoder_1", "sd3_text_encoder_2", "sd3_text_encoder_3", "sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    (None, "d9e0290829ba8d98e28e1a2b1407db4a", ["sd3_text_encoder_1"], [SD3TextEncoder1], "civitai"),
    # (None, "5072d0b24e406b49507abe861cf97691", ["sd3_text_encoder_3"], [SD3TextEncoder3], "civitai"),
    # (None, "4cf64a799d04260df438c6f33c9a047e", ["sdxl_text_encoder", "sdxl_text_encoder_2", "sdxl_unet", "sdxl_vae_decoder", "sdxl_vae_encoder"], [SDXLTextEncoder, SDXLTextEncoder2, SDXLUNet, SDXLVAEDecoder, SDXLVAEEncoder], "civitai"),
    # (None, "d9b008a867c498ab12ad24042eff8e3f", ["sdxl_text_encoder", "sdxl_text_encoder_2", "sdxl_unet", "sdxl_vae_decoder", "sdxl_vae_encoder"], [SDXLTextEncoder, SDXLTextEncoder2, SDXLUNet, SDXLVAEDecoder, SDXLVAEEncoder], "civitai"), # SDXL-Turbo
    # (None, "025bb7452e531a3853d951d77c63f032", ["sdxl_text_encoder", "sdxl_text_encoder_2"], [SDXLTextEncoder, SDXLTextEncoder2], "civitai"),
    # (None, "298997b403a4245c04102c9f36aac348", ["sdxl_unet"], [SDXLUNet], "civitai"),
    # (None, "2a07abce74b4bdc696b76254ab474da6", ["svd_image_encoder", "svd_unet", "svd_vae_decoder", "svd_vae_encoder"], [SVDImageEncoder, SVDUNet, SVDVAEDecoder, SVDVAEEncoder], "civitai"),
    # (None, "c96a285a6888465f87de22a984d049fb", ["sd_motion_modules"], [SDMotionModel], "civitai"),
    # (None, "72907b92caed19bdb2adb89aa4063fe2", ["sdxl_motion_modules"], [SDXLMotionModel], "civitai"),
    # (None, "31d2d9614fba60511fc9bf2604aa01f7", ["sdxl_controlnet"], [SDXLControlNetUnion], "diffusers"),
    (None, "94eefa3dac9cec93cb1ebaf1747d7b78", ["sd3_text_encoder_1"], [SD3TextEncoder1], "diffusers"),
    (None, "1aafa3cc91716fb6b300cc1cd51b85a3", ["flux_vae_encoder", "flux_vae_decoder"], [FluxVAEEncoder, FluxVAEDecoder], "diffusers"),
    (None, "21ea55f476dfc4fd135587abb59dfe5d", ["flux_vae_encoder", "flux_vae_decoder"], [FluxVAEEncoder, FluxVAEDecoder], "civitai"),
    (None, "a29710fea6dddb0314663ee823598e50", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "57b02550baab820169365b3ee3afa2c9", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "3394f306c4cbf04334b712bf5aaed95f", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "023f054d918a84ccf503481fd1e3379e", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "d02f41c13549fa5093d3521f62a5570a", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "605c56eab23e9e2af863ad8f0813a25d", ["flux_dit"], [FluxDiT], "diffusers"),
    # (None, "0629116fce1472503a66992f96f3eb1a", ["flux_value_controller"], [SingleValueEncoder], "civitai"),
    # (None, "280189ee084bca10f70907bf6ce1649d", ["cog_vae_encoder", "cog_vae_decoder"], [CogVAEEncoder, CogVAEDecoder], "diffusers"),
    # (None, "9b9313d104ac4df27991352fec013fd4", ["rife"], [IFNet], "civitai"),
    # (None, "6b7116078c4170bfbeaedc8fe71f6649", ["esrgan"], [RRDBNet], "civitai"),
    # (None, "61cbcbc7ac11f169c5949223efa960d1", ["omnigen_transformer"], [OmniGenTransformer], "diffusers"),
    # (None, "78d18b9101345ff695f312e7e62538c0", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    # (None, "b001c89139b5f053c715fe772362dd2a", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    # (None, "52357cb26250681367488a8954c271e8", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    # (None, "0cfd1740758423a2a854d67c136d1e8c", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    # (None, "7f9583eb8ba86642abb9a21a4b2c9e16", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    # (None, "43ad5aaa27dd4ee01b832ed16773fa52", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    # (None, "c07c0f04f5ff55e86b4e937c7a40d481", ["infiniteyou_image_projector"], [InfiniteYouImageProjector], "diffusers"),
    # (None, "4daaa66cc656a8fe369908693dad0a35", ["flux_ipadapter"], [FluxIpAdapter], "diffusers"),
    # (None, "51aed3d27d482fceb5e0739b03060e8f", ["sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    # (None, "98cc34ccc5b54ae0e56bdea8688dcd5a", ["sd3_text_encoder_2"], [SD3TextEncoder2], "civitai"),
    # (None, "77ff18050dbc23f50382e45d51a779fe", ["sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    (None, "5da81baee73198a7c19e6d2fe8b5148e", ["sd3_text_encoder_1"], [SD3TextEncoder1], "diffusers"),
    # (None, "aeb82dce778a03dcb4d726cb03f3c43f", ["hunyuan_video_vae_decoder", "hunyuan_video_vae_encoder"], [HunyuanVideoVAEDecoder, HunyuanVideoVAEEncoder], "diffusers"),
    # (None, "b9588f02e78f5ccafc9d7c0294e46308", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    # (None, "84ef4bd4757f60e906b54aa6a7815dc6", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    # (None, "68beaf8429b7c11aa8ca05b1bd0058bd", ["stepvideo_vae"], [StepVideoVAE], "civitai"),
    # (None, "5c0216a2132b082c10cb7a0e0377e681", ["stepvideo_dit"], [StepVideoModel], "civitai"),
    # (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"), 
    # (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"), 
    # (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    # (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    # (None, "7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    # (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    # (None, "966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),
    # (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    # (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    # (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    # (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    # (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
    # (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    (None, "d30fb9e02b1dbf4e509142f05cf7dd50", ["flux_dit"], [FluxDiT], "civitai"),
    # (None, "30143afb2dea73d1ac580e0787628f8c", ["flux_lora_patcher"], [FluxLoraPatcher], "civitai"),
    # (None, "77c2e4dd2440269eb33bfaa0d004f6ab", ["flux_lora_encoder"], [FluxLoRAEncoder], "civitai"),
    (None, "3e6c61b0f9471135fc9c6d6a98e98b6d", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "63c969fd37cce769a90aa781fbff5f81", ["flux_dit", "nexus_gen_editing_adapter"], [FluxDiT], "civitai"),
    # (None, "2bd19e845116e4f875a0a048e27fc219", ["nexus_gen_llm"], [NexusGenAutoregressiveModel], "civitai"),
    # (None, "0319a1cb19835fb510907dd3367c95ff", ["qwen_image_dit"], [QwenImageDiT], "civitai"),
    # (None, "8004730443f55db63092006dd9f7110e", ["qwen_image_text_encoder"], [QwenImageTextEncoder], "diffusers"),
    # (None, "ed4ea5824d55ec3107b09815e318123a", ["qwen_image_vae"], [QwenImageVAE], "diffusers"),
    # (None, "073bce9cf969e317e5662cd570c3e79c", ["qwen_image_blockwise_controlnet"], [QwenImageBlockWiseControlNet], "civitai"),
    # (None, "a9e54e480a628f0b956a688a81c33bab", ["qwen_image_blockwise_controlnet"], [QwenImageBlockWiseControlNet], "civitai"),
    # (None, "06be60f3a4526586d8431cd038a71486", ["wans2v_audio_encoder"], [WanS2VAudioEncoder], "civitai"),
]
huggingface_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (architecture_in_huggingface_config, huggingface_lib, model_name, redirected_architecture)
    ("ChatGLMModel", "models.kolors_text_encoder", "kolors_text_encoder", None),
    ("MarianMTModel", "transformers.models.marian.modeling_marian", "translator", None),
    ("BloomForCausalLM", "transformers.models.bloom.modeling_bloom", "beautiful_prompt", None),
    ("Qwen2ForCausalLM", "transformers.models.qwen2.modeling_qwen2", "qwen_prompt", None),
    # ("LlamaForCausalLM", "transformers.models.llama.modeling_llama", "omost_prompt", None),
    ("T5EncoderModel", "models.flux_text_encoder", "flux_text_encoder_2", "FluxTextEncoder2"),
    ("CogVideoXTransformer3DModel", "models.cog_dit", "cog_dit", "CogDiT"),
    ("SiglipModel", "transformers.models.siglip.modeling_siglip", "siglip_vision_model", "SiglipVisionModel"),
    ("LlamaForCausalLM", "models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoLLMEncoder"),
    ("LlavaForConditionalGeneration", "models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoMLLMEncoder"),
    ("Step1Model", "models.stepvideo_text_encoder", "stepvideo_text_encoder_2", "STEP1TextEncoder"),
    ("Qwen2_5_VLForConditionalGeneration", "models.qwenvl", "qwenvl", "Qwen25VL_7b_Embedder"),
    # ("CLIPModel", "transformers.models.clip.modeling_clip", "clip_model", None),
    ("CLIPTextModel", "transformers.models.clip.modeling_clip", "clip_text_encoder", None),
    ("AutoencoderKL", "diffusers.models.autoencoders.autoencoder_kl", "vae", None),
    ("AutoencoderKLQwenImage", "models.qwen_image_vae","qwen_image_vae", "QwenImageVAE"),
    ("FluxTransformer2DModel", "diffusers.models.transformers.transformer_flux", "flux_transformer2dmodel", None),
    ("CLIPTokenizer", "transformers.models.clip.tokenization_clip", "clip_tokenizer", None),
    ("T5Tokenizer", "transformers.models.t5.tokenization_t5", "t5_tokenizer", None),
    ("QwenImageTransformer2DModel", "models.qwen_image_dit", "qwen_image_dit", "QwenImageDiT"),
]
patch_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash_with_shape, model_name, model_class, extra_kwargs)
    # ("9a4ab6869ac9b7d6e31f9854e397c867", ["svd_unet"], [SVDUNet], {"add_positional_conv": 128}),
]

preset_models_on_huggingface = {}
preset_models_on_modelscope = {}
Preset_model_id: TypeAlias = Literal["local"]
