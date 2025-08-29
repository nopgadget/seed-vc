import torch
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor

class AstralQuantizer(torch.nn.Module):
    def __init__(
            self,
            tokenizer_name: str,
            ssl_model_name: str,
            ssl_output_layer: int,
            encoder: torch.nn.Module,
            quantizer: torch.nn.Module,
            decoder: torch.nn.Module = None,
            asr_decoder: torch.nn.Module = None,
            skip_ssl: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.asr_decoder = asr_decoder
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load SSL model from Huggingface
        self.ssl_model_name = ssl_model_name
        self.ssl_output_layer = ssl_output_layer
        self.ssl_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model_name)

        if skip_ssl:  # in case the same SSL model has been loaded somewhere else
            self.ssl_model = None
        else:
            self.ssl_model = AutoModel.from_pretrained(ssl_model_name).eval()
            self.ssl_model.encoder.layers = self.ssl_model.encoder.layers[:ssl_output_layer]
            self.ssl_model.encoder.layer_norm = torch.nn.Identity()

    def load_separate_checkpoint(self, checkpoint_path):
        params = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint structures
        if 'net' in params:
            # Original structure with 'net' wrapper
            params = params['net']
        
        # Separate parameters by component
        encoder_params = {}
        quantizer_params = {}
        decoder_params = {}
        asr_decoder_params = {}
        
        for key, value in params.items():
            if key.startswith("encoder."):
                # Remove "encoder." prefix
                encoder_params[key[8:]] = value
            elif key.startswith("quantizer."):
                # Remove "quantizer." prefix and fix corrupted keys
                clean_key = key[11:]
                
                # Fix corrupted keys
                if clean_key == "ask":
                    clean_key = "mask"
                elif clean_key == "roject_in.weight":
                    clean_key = "project_in.weight"
                elif clean_key == "roject_in.bias":
                    clean_key = "project_in.bias"
                elif clean_key == "roject_out.weight":
                    clean_key = "project_out.weight"
                elif clean_key == "roject_out.bias":
                    clean_key = "project_out.bias"
                elif clean_key == "project_in.weightt":
                    clean_key = "project_in.weight"
                elif clean_key == "project_out.biasias":
                    clean_key = "project_out.bias"
                elif clean_key == "project_out.weightght":
                    clean_key = "project_out.weight"
                
                quantizer_params[clean_key] = value
            elif key.startswith("decoder."):
                # Remove "decoder." prefix
                decoder_params[key[8:]] = value
            elif key.startswith("asr_decoder."):
                # Remove "asr_decoder." prefix
                asr_decoder_params[key[12:]] = value
        
        # Load encoder
        if encoder_params:
            self.encoder.load_state_dict(encoder_params)
        
        # Load quantizer
        if quantizer_params:
            self.quantizer.load_state_dict(quantizer_params)
        
        # Load decoder if exists
        if hasattr(self, 'decoder') and self.decoder is not None and decoder_params:
            self.decoder.load_state_dict(decoder_params)
        
        # Load ASR decoder if exists
        if hasattr(self, 'asr_decoder') and self.asr_decoder is not None and asr_decoder_params:
            self.asr_decoder.load_state_dict(asr_decoder_params, strict=False)

    def forward(self, waves_16k, wave_16k_lens, ssl_model=None):
        ssl_fn = self.ssl_model if self.ssl_model else ssl_model
        assert ssl_fn is not None, "In case in-class SSL model loading is skipped, external ssl_model must be provided"
        waves_16k_input_list = [
            waves_16k[bib, :wave_16k_lens[bib]].cpu().numpy()
            for bib in range(len(waves_16k))
        ]
        alt_inputs = self.ssl_feature_extractor(
            waves_16k_input_list,
            return_tensors='pt',
            return_attention_mask=True,
            padding=True,
            sampling_rate=16000
        ).to(waves_16k.device)
        feature_lens = alt_inputs.data['attention_mask'].sum(-1) // 320  # frame rate of hubert is 50 Hz

        outputs = ssl_fn(
            alt_inputs.input_values,
            attention_mask=alt_inputs.attention_mask,
        )
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states[:, :feature_lens.max(), :]
        feature_lens = feature_lens.clamp(max=last_hidden_states.size(1))
        last_hidden_states = last_hidden_states.transpose(1, 2)
        x_hidden = self.encoder(last_hidden_states, feature_lens)
        x_hidden = x_hidden.transpose(1, 2)
        x_quantized, indices = self.quantizer(x_hidden)[:2]
        return x_quantized, indices, feature_lens