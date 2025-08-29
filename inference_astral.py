#!/usr/bin/env python3
"""
Simple inference script for ASTRAL Quantization voice conversion
This script tests the complete pipeline with a simple audio file.
"""

import torch
import torchaudio
import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.getcwd())

def load_astral_models(config_path, device="cuda"):
    """Load ASTRAL quantization models"""
    try:
        from modules.astral_quantization.default_model import AstralQuantizer
        from modules.astral_quantization.bsq import BSQQuantizer
        from modules.astral_quantization.convnext import ConvNeXtEncoder
        from modules.astral_quantization.asr_decoder import ASRDecoder
        import yaml
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get ASTRAL config path
        astral_config_path = config['model_params']['speech_tokenizer']['config_path']
        astral_checkpoint_path = config['model_params']['speech_tokenizer']['checkpoint_path']
        
        # Load ASTRAL configuration
        with open(astral_config_path, 'r') as f:
            astral_config = yaml.safe_load(f)
        
        # Create model components
        encoder = ConvNeXtEncoder(**astral_config['encoder'])
        quantizer = BSQQuantizer(**astral_config['quantizer'])
        decoder = ASRDecoder(**astral_config['decoder']) if 'decoder' in astral_config else None
        asr_decoder = ASRDecoder(**astral_config['asr_decoder']) if 'asr_decoder' in astral_config else None
        
        # Create ASTRAL model
        astral_model = AstralQuantizer(
            tokenizer_name=astral_config['tokenizer_name'],
            ssl_model_name=astral_config['ssl_model_name'],
            ssl_output_layer=astral_config['ssl_output_layer'],
            encoder=encoder,
            quantizer=quantizer,
            decoder=decoder,
            asr_decoder=asr_decoder,
            skip_ssl=False
        )
        
        # Load checkpoint
        astral_model.load_separate_checkpoint(astral_checkpoint_path)
        astral_model.eval()
        astral_model.to(device)
        
        # Create projection layer once
        projection_layer = torch.nn.Linear(384, 1024).to(device)
        
        print(f"✅ ASTRAL model loaded from: {astral_checkpoint_path}")
        return astral_model, projection_layer, config
        
    except Exception as e:
        print(f"❌ Failed to load ASTRAL models: {e}")
        return None, None, None

def process_audio(audio_path, astral_model, projection_layer, device="cuda"):
    """Process audio through ASTRAL quantization"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Move to device
        waveform = waveform.to(device)
        
        # Calculate lengths
        wave_lengths = torch.LongTensor([waveform.size(-1)]).to(device)
        
        print(f"📊 Audio loaded: {waveform.shape}, {sample_rate}Hz")
        
        with torch.no_grad():
            # Get ASTRAL features
            x_quantized, indices, feature_lens = astral_model(waveform, wave_lengths)
            
            print(f"✅ ASTRAL output: {x_quantized.shape}")
            
            # Project dimensions if needed
            if x_quantized.size(-1) == 384:
                x_quantized = projection_layer(x_quantized)
                print(f"✅ Projected output: {x_quantized.shape}")
            
            return x_quantized, indices, feature_lens
            
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Test ASTRAL Quantization inference")
        parser.add_argument("--config", type=str, 
                        default="configs/astral_quantization/config_dit_astral_bsq32.yml",
                        help="Configuration file path")
    parser.add_argument("--audio", type=str, required=True,
                       help="Input audio file path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--output", type=str, default="output/converted_audio/astral_output.wav",
                       help="Output audio file path")
    
    args = parser.parse_args()
    
    print("🚀 ASTRAL Quantization Inference Test")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        return
    
    # Load models
    print("📥 Loading ASTRAL models...")
    astral_model, projection_layer, config = load_astral_models(args.config, args.device)
    
    if astral_model is None:
        print("❌ Failed to load models")
        return
    
    # Process audio
    print(f"🎵 Processing audio: {args.audio}")
    x_quantized, indices, feature_lens = process_audio(args.audio, astral_model, projection_layer, args.device)
    
    if x_quantized is None:
        print("❌ Failed to process audio")
        return
    
    print("✅ Audio processing completed successfully!")
    print(f"📊 Final output shape: {x_quantized.shape}")
    print(f"📊 Indices shape: {indices.shape}")
    print(f"📊 Feature lengths: {feature_lens}")
    
    # Save processed features (for debugging)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "astral_features").mkdir(exist_ok=True)
    (output_dir / "converted_audio").mkdir(exist_ok=True)
    
    torch.save({
        'x_quantized': x_quantized.cpu(),
        'indices': indices.cpu(),
        'feature_lens': feature_lens.cpu(),
        'config': config
    }, output_dir / "astral_features" / "astral_features.pt")
    
    print(f"💾 Features saved to: {output_dir / 'astral_features' / 'astral_features.pt'}")
    print("🎉 ASTRAL quantization inference test completed successfully!")

if __name__ == "__main__":
    main()
