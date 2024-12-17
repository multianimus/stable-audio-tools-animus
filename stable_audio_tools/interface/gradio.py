from asyncio.windows_events import NULL
import gc
import os

import ffmpeg
import threading
import numpy as np
import gradio as gr
import json 
import torch
import torchaudio
import threading
import time
import re

import hffs

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T
from pydub import AudioSegment

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict

import matplotlib.pyplot as plt
import librosa.display

# Load config file
with open("config.json") as config_file:
    config = json.load(config_file)

model = None
sample_rate = 32000
sample_size = 1920000
DEVICE = None
global_model_half = False

output_directory = config['generations_directory']

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

generate_forever_flag = False


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device=None):
    global model, sample_rate, sample_size, global_model_half
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)
        
        # Load checkpoint
        state_dict = load_ckpt_state_dict(model_ckpt_path)
        
        # Check if the model is in float16 format before loading into the model
        is_float16 = all(param.dtype == torch.float16 for param in state_dict.values())
        
        if global_model_half or is_float16 :
            print("Model is in float16 format. Enabling half-precision inference.")
            global_model_half = True
            model.to(torch.float16)  # Convert the model to half precision before loading state dict
        else:
            print("Model is in full precision format.")
            global_model_half = False
        
        model.load_state_dict(state_dict)
        
        # Print parameter types after loading into the model
        #print("Parameter types after loading into the model:")
        #for name, param in model.named_parameters():
        #    print(f"Parameter {name} has dtype {param.dtype}")
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        pretransform_state_dict = load_ckpt_state_dict(pretransform_ckpt_path)
        
        # Check if the pretransform model is in float16 format before loading into the pretransform model
        is_float16_pretransform = all(param.dtype == torch.float16 for param in pretransform_state_dict.values())
        
        if is_float16_pretransform:
            print("Model is in float16 format. Enabling half-precision inference.")
            model.pretransform.to(torch.float16)  # Convert the pretransform model to half precision before loading state dict
        else:
            print("Model is in full precision format.")
        
        model.pretransform.load_state_dict(pretransform_state_dict, strict=False)
        #print(f"Done loading pretransform")
    
    # Move the model to the specified device
    model.to(device).eval().requires_grad_(False)
    
    print(f"Done loading model")
    return model, model_config

def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1    
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Return fake stereo audio
    conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size

    if negative_prompt:
        negative_conditioning = [{"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size
    else:
        negative_conditioning = None
        
    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # If inpainting, send mask args
    # This will definitely change in the future
    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 

    # Do the audio generation
    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=input_sample_size,
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        mask_args = mask_args,
        callback = progress_callback if preview_every is not None else None,
        scale_phi = cfg_rescale
    )

    # Convert to WAV file (temporary file)
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save("temp_output.wav", audio, sample_rate)

    # Trim audio using ffmpeg
    trim_audio("temp_output.wav", "output.wav", seconds_total)

    # Let's look at a nice spectrogram too
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram, *preview_images])

def trim_audio(input_file, output_file, duration_seconds):
    stream = ffmpeg.input(input_file)
    audio_stream = stream.audio
    trimmed = audio_stream.filter('atrim', end=duration_seconds)
    output = ffmpeg.output(trimmed, output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    ffmpeg.run(output)
    os.remove(input_file) # removes the temp file
    return

def generate_uncond(
        steps=250,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        batch_size=1,
        preview_every=None
        ):

    global preview_images

    preview_images = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:

            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)

            denoised = rearrange(denoised, "b d n -> d (b n)")

            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)

            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    audio = generate_diffusion_uncond(
        model, 
        steps=steps,
        batch_size=batch_size,
        sample_size=input_sample_size,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        callback = progress_callback if preview_every is not None else None
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram, *preview_images])

def generate_lm(
        temperature=1.0,
        top_p=0.95,
        top_k=0,    
        batch_size=1,
        ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    audio = model.generate_audio(
        batch_size=batch_size,
        max_gen_len = sample_size//model.pretransform.downsampling_ratio,
        conditioning=None,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram])


def create_uncond_sampling_ui(model_config):   
    generate_button = gr.Button("Generate", variant='primary', scale=1)
    
    with gr.Row(equal_height=False):
        with gr.Column():            
            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")

            with gr.Accordion("Sampler params", open=False):
            
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

            # Sampler params
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")

            with gr.Accordion("Init audio", open=False):
                init_audio_checkbox = gr.Checkbox(label="Use init audio")
                init_audio_input = gr.Audio(label="Init audio")
                init_noise_level_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])
    
    generate_button.click(fn=generate_uncond, 
        inputs=[
            steps_slider, 
            seed_textbox, 
            sampler_type_dropdown, 
            sigma_min_slider, 
            sigma_max_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
        ], 
        outputs=[
            audio_output, 
            audio_spectrogram_output
        ], 
        api_name="generate")

def load_model_action(selected_ckpt, selected_config, ckpt_files):
    global DEVICE, current_prompt_generator
    try:
        ckpt_path = next(path for name, path in ckpt_files if name == selected_ckpt)
        config_path = os.path.join(os.path.dirname(ckpt_path), selected_config)
        
        model, model_config = load_model(
            model_config=json.load(open(config_path)),
            model_ckpt_path=ckpt_path,
            device=DEVICE
        )
        
        return f"Loaded model {selected_ckpt} with config {selected_config}"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error loading model: {e}"
    
def create_txt2audio_ui(model_config, initial_ckpt):
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_config, initial_ckpt)
        with gr.Tab("Inpainting"):
            create_sampling_ui(model_config, initial_ckpt, inpainting=True)
        with gr.Tab("Download Models"):
            gr.HTML("<h2>Download</h2><div>Download a model and restart the app to apply.</div>")
            hffs.from_config(config)
            
    return ui

def create_diffusion_uncond_ui(model_config):
    with gr.Blocks() as ui:
        create_uncond_sampling_ui(model_config)
    
    return ui

def autoencoder_process(audio, latent_noise, n_quantizers):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM, 
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model. 
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    if n_quantizers > 0:
        latents = model.encode_audio(audio, chunked=False, n_quantizers=n_quantizers)
    else:
        latents = model.encode_audio(audio, chunked=False)

    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_autoencoder_ui(model_config):

    is_dac_rvq = "model" in model_config and "bottleneck" in model_config["model"] and model_config["model"]["bottleneck"]["type"] in ["dac_rvq","dac_rvq_vae"]

    if is_dac_rvq:
        n_quantizers = model_config["model"]["bottleneck"]["config"]["n_codebooks"]
    else:
        n_quantizers = 0

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        n_quantizers_slider = gr.Slider(minimum=1, maximum=n_quantizers, step=1, value=n_quantizers, label="# quantizers", visible=is_dac_rvq)
        latent_noise_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.001, value=0.0, label="Add latent noise")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=autoencoder_process, inputs=[input_audio, latent_noise_slider, n_quantizers_slider], outputs=output_audio, api_name="process")

    return ui

def diffusion_prior_process(audio, steps, sampler_type, sigma_min, sigma_max):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    audio = audio.unsqueeze(0)

    audio = model.stereoize(audio, in_sr, steps, sampler_kwargs={"sampler_type": sampler_type, "sigma_min": sigma_min, "sigma_max": sigma_max})

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

# Updated generate_cond_with_filename to ensure it uses the filename properly
def generate_cond_with_filename(
        filename,
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1    
    ):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    if not filename:
        filename = ensure_unique_filename(output_folder, ensure_wav_extension(sanitize_filename(prompt)))
    else:
        filename = ensure_unique_filename(output_folder, ensure_wav_extension(filename))
    
    # Call the original generation function
    audio_path, spectrograms = generate_cond(
        prompt,
        negative_prompt,
        seconds_start,
        seconds_total,
        cfg_scale,
        steps,
        preview_every,
        seed,
        sampler_type,
        sigma_min,
        sigma_max,
        cfg_rescale,
        use_init,
        init_audio,
        init_noise_level,
        mask_cropfrom,
        mask_pastefrom,
        mask_pasteto,
        mask_maskstart,
        mask_maskend,
        mask_softnessL,
        mask_softnessR,
        mask_marination,
        batch_size
    )

    # Save to the unique file
    final_path = os.path.join(output_folder, filename)
    os.rename(audio_path, final_path)

    return final_path, spectrograms

def generate_forever(
    filename,
    prompt,
    negative_prompt=None,
    seconds_start=0,
    seconds_total=30,
    cfg_scale=6.0,
    steps=250,
    preview_every=None,
    seed=-1,
    sampler_type="dpmpp-3m-sde",
    sigma_min=0.03,
    sigma_max=1000,
    cfg_rescale=0.0,
    use_init=False,
    init_audio=None,
    init_noise_level=1.0,
    mask_cropfrom=None,
    mask_pastefrom=None,
    mask_pasteto=None,
    mask_maskstart=None,
    mask_maskend=None,
    mask_softnessL=None,
    mask_softnessR=None,
    mask_marination=None,
    batch_size=1    
):
    def loop_forever():
        global generate_forever_flag
        
        generate_forever_flag = True
        # Start infinite loop if the flag is True
        while generate_forever_flag:
            print("Generating...")  # You can replace this with actual generation logic
            # Call the generate_cond_with_filename function with the provided parameters
            generate_cond_with_filename(
                filename,
                prompt,
                negative_prompt,
                seconds_start,
                seconds_total,
                cfg_scale,
                steps,
                preview_every,
                seed,
                sampler_type,
                sigma_min,
                sigma_max,
                cfg_rescale,
                use_init,
                init_audio,
                init_noise_level,
                mask_cropfrom,
                mask_pastefrom,
                mask_pasteto,
                mask_maskstart,
                mask_maskend,
                mask_softnessL,
                mask_softnessR,
                mask_marination,
                batch_size
            )
            time.sleep(1)  # Simulate delay (e.g., for time-consuming tasks like audio generation)

        print("Generation stopped.")  # Log when the loop is stopped

    # Start the infinite generation loop in a separate thread
    threading.Thread(target=loop_forever, daemon=True).start()

def stop_generation():
    global generate_forever_flag
    # Set the flag to False to stop the infinite generation loop
    generate_forever_flag = False
    print("Generation process stopped.")

# Updated UI Function
def create_sampling_ui(model_config, initial_ckpt, inpainting=False):
    ckpt_files = get_models_and_configs(config['models_directory'])
    
    if initial_ckpt == "none":
        selected_ckpt = "none"
    else:
        selected_ckpt = gr.State(value=os.path.basename(initial_ckpt))
        
    selected_config = gr.State()

    with gr.Blocks() as ui:
      
      with gr.Row():
        with gr.Column(scale=8):  # Input fields take more space
            prompt = gr.Textbox(show_label=False, placeholder="Prompt")
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
            filename_textbox = gr.Textbox(label="Filename", placeholder="Enter filename for output (optional)")
        with gr.Column(scale=2):  # Buttons take less space
            with gr.Column():
                generate_button = gr.Button("Generate", variant='primary', scale=1)
                generate_forever_button = gr.Button("Generate Forever", variant="secondary", scale=1)
                stop_button = gr.Button("Cancel Forever", variant="danger", scale=1)
            
    model_conditioning_config = model_config["model"].get("conditioning", None)

    has_seconds_start = False
    has_seconds_total = False

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True

    # Initialize `init_audio_input` outside conditional blocks
    #init_audio_input = None

    with gr.Row(equal_height=False):
        with gr.Column():
            
            if selected_ckpt  == "none":
                current_model_info = gr.Markdown(f"Current Model: None")
            else:
                current_model_info = gr.Markdown(f"Current Model: {selected_ckpt.value}")
                
            # Comment out the model and config dropdowns / load model button for demos
            with gr.Row():
                # Model and Config dropdowns
                model_dropdown = gr.Dropdown(["Select Model"] + [file[0] for file in ckpt_files], label="Select Model")
                config_dropdown = gr.Dropdown(["Select Config"], label="Select Config")
            
            model_dropdown.change(fn=lambda x: update_config_dropdown(x, ckpt_files), inputs=model_dropdown, outputs=config_dropdown)

            load_model_button = gr.Button("Load Model")
            
            with gr.Row(visible=has_seconds_start or has_seconds_total):
                # Timing controls
                seconds_start_slider = gr.Slider(
                    minimum=0, maximum=512, step=1, value=0, label="Seconds start", visible=has_seconds_start
                )
                seconds_total_slider = gr.Slider(
                    minimum=1, maximum=47, step=1, value=30, label="Seconds total", visible=has_seconds_total
                )

            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")

                # Preview slider
                preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")

                # CFG scale
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG Scale")

            # Advanced settings in an accordion
            with gr.Accordion("Sampler params", open=False):
                # Seed input
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                # Sampler options
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(
                        ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                        label="Sampler Type",
                        value="dpmpp-3m-sde",
                    )
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma Min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500.0, label="Sigma Max")
                    cfg_rescale_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="CFG Rescale Amount"
                    )

            # Init audio options for inpainting
            if inpainting:
                with gr.Accordion("Inpainting", open=False):
                    init_audio_checkbox = gr.Checkbox(label="Use Init Audio")
                    init_audio_input = gr.Audio(label="Init Audio")  # Defined here for inpainting mode
                    init_noise_level_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.01, value=0.1, label="Init Noise Level"
                    )

                    # Inpainting-specific sliders
                    mask_cropfrom_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.1, value=0.0, label="Crop From %"
                    )
                    mask_pastefrom_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.1, value=0.0, label="Paste From %"
                    )
                    mask_pasteto_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.1, value=100.0, label="Paste To %"
                    )
                    mask_maskstart_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.1, value=50.0, label="Mask Start %"
                    )
                    mask_maskend_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.1, value=100.0, label="Mask End %"
                    )
                    mask_softnessL_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.1, value=0.0, label="Softmask Left Crossfade Length %"
                    )
                    mask_softnessR_slider = gr.Slider(
                        minimum=0.0, maximum=100.0, step=0.1, value=0.0, label="Softmask Right Crossfade Length %"
                    )
                    mask_marination_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.001, value=0.0, label="Marination Level"
                    )
            else:
                        # Default generation tab
                with gr.Accordion("Init audio", open=False):
                    init_audio_checkbox = gr.Checkbox(label="Use init audio")
                    init_audio_input = gr.Audio(label="Init audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

        with gr.Column():
            # Outputs
            audio_output = gr.Audio(label="Generated Audio", interactive=False)
            spectrogram_output = gr.Gallery(label="Spectrogram", show_label=False)

            # Send to Init Audio Button
            if not inpainting:
                init_audio_input = gr.Audio(label="Init Audio")  # Define if not inpainting
            send_to_init_button = gr.Button("Send to Init Audio")
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])
    
    if inpainting:
        inputs = [
                        prompt,
                        negative_prompt,
                        seconds_start_slider,
                        seconds_total_slider,
                        cfg_scale_slider,
                        steps_slider,
                        preview_every_slider,
                        seed_textbox,
                        sampler_type_dropdown,
                        sigma_min_slider,
                        sigma_max_slider,
                        cfg_rescale_slider,
                        init_audio_checkbox,
                        init_audio_input,
                        init_noise_level_slider,
                        mask_cropfrom_slider,
                        mask_pastefrom_slider,
                        mask_pasteto_slider,
                        mask_maskstart_slider,
                        mask_maskend_slider,
                        mask_softnessL_slider,
                        mask_softnessR_slider,
                        mask_marination_slider,
                    ]
    else:
         inputs = [prompt, 
                        negative_prompt,
                        seconds_start_slider, 
                        seconds_total_slider, 
                        cfg_scale_slider, 
                        steps_slider, 
                        preview_every_slider, 
                        seed_textbox, 
                        sampler_type_dropdown, 
                        sigma_min_slider, 
                        sigma_max_slider,
                        cfg_rescale_slider,
                        init_audio_checkbox,
                        init_audio_input,
                        init_noise_level_slider
                    ]

    # Generate button click
    generate_button.click(
        fn=generate_cond_with_filename,
        inputs=[filename_textbox, *inputs],
        outputs=[audio_output, spectrogram_output],
    )

        # Generate Forever button starts the infinite generation
    generate_forever_button.click(
        fn=generate_forever,
        inputs=[filename_textbox, *inputs],
        outputs=[],
    )
    # Stop button will toggle off the infinite generation
    stop_button.click(fn=stop_generation, inputs=[], outputs=[])
    
    send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

    # Comment out the load model button click event
    load_model_button.click(fn=lambda x, y: load_model_action(x, y, ckpt_files), inputs=[model_dropdown, config_dropdown], outputs=[current_model_info])

    return ui

def update_config_dropdown(selected_ckpt, ckpt_files):
    try:
        ckpt_path = next(path for name, path in ckpt_files if name == selected_ckpt)
        configs = get_config_files(ckpt_path)
        return gr.update(choices=configs, value=configs[0] if configs else "Select Config")
    except Exception as e:
        print(f"Error updating config dropdown: {e}")  # Debugging output
        return gr.update(choices=["Error finding configs"], value="Error finding configs")
    
def ensure_wav_extension(filename):
    """Ensure the filename ends with .wav."""
    if not filename.endswith(".wav"):
        filename += ".wav"
    return filename

def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Removes invalid characters from a string to make it a valid filename.

    Args:
        filename (str): The original filename.
        replacement (str): The character to replace invalid characters with (default is "_").

    Returns:
        str: The sanitized filename.
    """
    # Define a regex for invalid characters (allow only alphanumeric, dash, underscore, and space)
    invalid_characters_pattern = r'[<>:"/\\|?*\x00-\x1F]'  # Invalid on most file systems
    sanitized = re.sub(invalid_characters_pattern, replacement, filename)
    
    # Remove leading and trailing spaces or dots, which are also invalid for filenames
    sanitized = sanitized.strip(" .")
    
    return sanitized 

def stop_generation(*args):
    global generate_forever_flag
    # Set the flag to False to stop the infinite generation
    
    if generate_forever_flag == False:
        print("Stopping generate loop")

    generate_forever_flag = False
    
    
def ensure_unique_filename(output_folder, filename):
    """Ensure the filename is unique by appending a number if it already exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    while os.path.exists(os.path.join(output_folder, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    return unique_filename


def create_diffusion_prior_ui(model_config):
    with gr.Blocks() as ui:
        # Input and output audio
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)

        # Sampler parameters
        with gr.Row():
            steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
            sampler_type_dropdown = gr.Dropdown(
                ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                label="Sampler type",
                value="dpmpp-3m-sde",
            )
            sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
            sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")

            # Textbox for filename added to the same row
            filename_textbox = gr.Textbox(label="Filename", placeholder="Enter filename here", elem_id="filename_box")

        # Process button
        process_button = gr.Button("Process", variant="primary", scale=1)

        # Click event for the button
        process_button.click(
            fn=diffusion_prior_process,
            inputs=[input_audio, filename_textbox, steps_slider, sampler_type_dropdown, sigma_min_slider, sigma_max_slider],
            outputs=output_audio,
            api_name="process",
        )

    return ui


def create_lm_ui(model_config):
    with gr.Blocks() as ui:
        output_audio = gr.Audio(label="Output audio", interactive=False)
        audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)

        # Sampling params
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0, maximum=5, step=0.01, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95, label="Top p")
            top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top k")

        generate_button = gr.Button("Generate", variant='primary', scale=1)
        generate_button.click(
            fn=generate_lm, 
            inputs=[
                temperature_slider, 
                top_p_slider, 
                top_k_slider
            ], 
            outputs=[output_audio, audio_spectrogram_output],
            api_name="generate"
        )

    return ui

def get_models_and_configs(models_path):
    ckpt_files = []
    for root, _, files in os.walk(models_path):
        for file in files:
            if file.endswith((".ckpt", ".safetensors")):
                ckpt_files.append((file, os.path.join(root, file)))
    return ckpt_files

def get_config_files(ckpt_path):
    config_files = []
    folder = os.path.dirname(ckpt_path)
    print(f"Looking for config files in folder: {folder}")  # Debugging output
    for file in os.listdir(folder):
        if file.endswith(".json"):
            config_files.append(file)
    print(f"Found config files: {config_files}")  # Debugging output
    return config_files

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    global global_model_half
    global current_prompt_generator
    global_model_half = model_half  # Initialize model_half with the provided value

    if pretrained_name is None and model_config_path is None and ckpt_path is None:
        print("checking the models folder for a default checkpoint")
        try:
            ckpt_files = get_models_and_configs(config['models_directory'])
            ckpt_path = ckpt_files[0][1]
            configs = get_config_files(ckpt_path)
            model_config_path = os.path.join(os.path.dirname(ckpt_path), configs[0])
        except IndexError:
            print("no default checkpoint.")
            with gr.Blocks() as ui:
                gr.HTML("<h2>Initialize</h2><div>Download a model first, and restart the app.</div>")
                hffs.from_config(config)
            return ui
    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    global DEVICE
    DEVICE = device

    #initial_ckpt = ckpt_path if ckpt_path is not None else pretrained_name
    initial_ckpt = "none"
    
    #_, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, device=device)

    model_type = model_config["model_type"]

    if model_type == "diffusion_cond":
        ui = create_txt2audio_ui(model_config, initial_ckpt)
    elif model_type == "diffusion_uncond":
        ui = create_diffusion_uncond_ui(model_config)
    elif model_type == "autoencoder" or model_type == "diffusion_autoencoder":
        ui = create_autoencoder_ui(model_config)
    elif model_type == "diffusion_prior":
        ui = create_diffusion_prior_ui(model_config)
    elif model_type == "lm":
        ui = create_lm_ui(model_config)

    return ui