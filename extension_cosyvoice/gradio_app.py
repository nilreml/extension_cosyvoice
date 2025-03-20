import os

import gradio as gr
import subprocess

# Global model cache
cosyvoice_model = None


def get_cosyvoice_model(
    model_name="CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=False
):
    global cosyvoice_model
    from cosyvoice.cli.cosyvoice import CosyVoice2

    if cosyvoice_model is None:
        models_dir = "data/models/cosyvoice"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_name)

        if not os.path.exists(model_path):
            gr.Info(f"Downloading {model_name} model...")
            from modelscope import snapshot_download

            snapshot_download(f"iic/{model_name}", local_dir=model_path)

        gr.Info(f"Loading {model_name} model...")
        cosyvoice_model = CosyVoice2(
            model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16
        )
        gr.Info(f"Model {model_name} loaded successfully")

    return cosyvoice_model


def load_prompt_audio(prompt_audio):
    # Deduplicate audio loading logic
    from cosyvoice.utils.file_utils import load_wav  # Ensure proper import

    return (
        load_wav(prompt_audio, 16000) if isinstance(prompt_audio, str) else prompt_audio
    )


def process_results(generation_results, sample_rate):
    # Deduplicate result iteration logic
    if not generation_results:
        return None
    import torch

    if len(generation_results) > 1:
        all_audio = torch.cat(
            [result["tts_speech"] for result in generation_results], dim=1
        )
        return (sample_rate, all_audio[0].cpu().numpy())
    else:
        return (sample_rate, generation_results[0]["tts_speech"][0].cpu().numpy())


def inference_zero_shot(
    prompt_audio, prompt_text, gen_text, stream=False, model_name="CosyVoice2-0.5B"
):
    model = get_cosyvoice_model(model_name)
    prompt_speech = load_prompt_audio(prompt_audio)
    generation_results = list(
        model.inference_zero_shot(gen_text, prompt_text, prompt_speech, stream=stream)
    )
    return process_results(generation_results, model.sample_rate)


def inference_instruct(
    prompt_audio,
    prompt_text,
    gen_text,
    instruct_text,
    stream=False,
    model_name="CosyVoice2-0.5B",
):
    model = get_cosyvoice_model(model_name)
    prompt_speech = load_prompt_audio(prompt_audio)
    generation_results = list(
        model.inference_instruct2(gen_text, instruct_text, prompt_speech, stream=stream)
    )
    return process_results(generation_results, model.sample_rate)


def inference_cross_lingual(
    prompt_audio, prompt_text, gen_text, stream=False, model_name="CosyVoice2-0.5B"
):
    model = get_cosyvoice_model(model_name)
    prompt_speech = load_prompt_audio(prompt_audio)
    generation_results = list(
        model.inference_cross_lingual(gen_text, prompt_speech, stream=stream)
    )
    return process_results(generation_results, model.sample_rate)


def inference_text_generator(
    prompt_audio,
    prompt_text,
    gen_text,
    stream=False,
    model_name="CosyVoice2-0.5B",
    chunk_size=100,
):
    model = get_cosyvoice_model(model_name)
    prompt_speech = load_prompt_audio(prompt_audio)

    def text_generator():
        words = gen_text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                yield chunk

    generation_results = list(
        model.inference_zero_shot(
            text_generator(), prompt_text, prompt_speech, stream=stream
        )
    )
    return process_results(generation_results, model.sample_rate)


def ui_app_tts():
    with gr.Blocks() as app_tts:
        gr.Markdown("# Zero-Shot TTS")
        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
        ref_text_input = gr.Textbox(
            label="Reference Text",
            lines=2,
            placeholder="Text that corresponds to the reference audio",
        )
        gen_text_input = gr.Textbox(label="Text to Generate", lines=10)

        with gr.Accordion("Advanced Settings", open=False):
            model_choice = gr.Radio(
                choices=[
                    "CosyVoice2-0.5B",
                    "CosyVoice-300M",
                    "CosyVoice-300M-25Hz",
                    "CosyVoice-300M-SFT",
                    "CosyVoice-300M-Instruct",
                    "CosyVoice-ttsfrd",
                ],
                label="Choose TTS Model",
                value="CosyVoice2-0.5B",
            )
            stream_checkbox = gr.Checkbox(label="Stream Output", value=False)

        generate_btn = gr.Button("Synthesize", variant="primary")
        audio_output = gr.Audio(label="Synthesized Audio")

        generate_btn.click(
            inference_zero_shot,
            inputs=[
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                stream_checkbox,
                model_choice,
            ],
            outputs=audio_output,
        )


def ui_app_instruct():
    with gr.Blocks() as app_instruct:
        gr.Markdown("# Instruct TTS")
        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
        ref_text_input = gr.Textbox(
            label="Reference Text",
            lines=2,
            placeholder="Text that corresponds to the reference audio",
        )
        gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
        instruct_text_input = gr.Textbox(
            label="Instruction",
            lines=2,
            placeholder="Instructions for text generation, e.g., 'use a Scottish accent', 'speak happily', etc.",
        )

        with gr.Accordion("Advanced Settings", open=False):
            model_choice = gr.Radio(
                choices=["CosyVoice2-0.5B"],
                label="Choose TTS Model",
                value="CosyVoice2-0.5B",
            )
            stream_checkbox = gr.Checkbox(label="Stream Output", value=False)

        generate_btn = gr.Button("Synthesize with Instructions", variant="primary")
        audio_output = gr.Audio(label="Synthesized Audio")

        generate_btn.click(
            inference_instruct,
            inputs=[
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                instruct_text_input,
                stream_checkbox,
                model_choice,
            ],
            outputs=audio_output,
        )


def ui_app_cross_lingual():
    with gr.Blocks() as app_cross:
        gr.Markdown("# Cross-Lingual TTS")
        gr.Markdown(
            "Use special tags like [laughter] to control fine-grained speech characteristics"
        )
        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
        ref_text_input = gr.Textbox(
            label="Reference Text",
            lines=2,
            placeholder="Text that corresponds to the reference audio",
        )
        gen_text_input = gr.Textbox(
            label="Text to Generate",
            lines=10,
            placeholder="Use tags like [laughter] to control aspects of speech, e.g.: 'He told a joke and [laughter] it was so funny'",
        )

        with gr.Accordion("Advanced Settings", open=False):
            model_choice = gr.Radio(
                choices=["CosyVoice2-0.5B"],
                label="Choose TTS Model",
                value="CosyVoice2-0.5B",
            )
            stream_checkbox = gr.Checkbox(label="Stream Output", value=False)

        generate_btn = gr.Button("Synthesize Cross-Lingual", variant="primary")
        audio_output = gr.Audio(label="Synthesized Audio")

        generate_btn.click(
            inference_cross_lingual,
            inputs=[
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                stream_checkbox,
                model_choice,
            ],
            outputs=audio_output,
        )


def ui_app_chunk():
    with gr.Blocks() as app_chunk:
        gr.Markdown("# Chunked Text TTS")
        gr.Markdown("For handling long text that will be broken into manageable chunks")
        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
        ref_text_input = gr.Textbox(
            label="Reference Text",
            lines=2,
            placeholder="Text that corresponds to the reference audio",
        )
        gen_text_input = gr.Textbox(
            label="Long Text to Generate",
            lines=15,
            placeholder="Enter a long text that will be automatically chunked",
        )

        with gr.Accordion("Advanced Settings", open=False):
            model_choice = gr.Radio(
                choices=["CosyVoice2-0.5B"],
                label="Choose TTS Model",
                value="CosyVoice2-0.5B",
            )
            chunk_size = gr.Slider(
                minimum=10, maximum=200, value=50, step=10, label="Words per Chunk"
            )
            stream_checkbox = gr.Checkbox(label="Stream Output", value=False)

        generate_btn = gr.Button("Synthesize Chunked Text", variant="primary")
        audio_output = gr.Audio(label="Synthesized Audio")

        def inference_chunked(
            prompt_audio, prompt_text, gen_text, stream, model_name, chunk_size
        ):
            return inference_text_generator(
                prompt_audio, prompt_text, gen_text, stream, model_name, int(chunk_size)
            )

        generate_btn.click(
            inference_chunked,
            inputs=[
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                stream_checkbox,
                model_choice,
                chunk_size,
            ],
            outputs=audio_output,
        )


def download_models(selected_models):
    from modelscope import snapshot_download
    import os

    model_mapping = {
        "CosyVoice2-0.5B": (
            "iic/CosyVoice2-0.5B",
            os.path.join("data", "models", "cosyvoice", "CosyVoice2-0.5B"),
        ),
        "CosyVoice-300M": (
            "iic/CosyVoice-300M",
            os.path.join("data", "models", "cosyvoice", "CosyVoice-300M"),
        ),
        "CosyVoice-300M-25Hz": (
            "iic/CosyVoice-300M-25Hz",
            os.path.join("data", "models", "cosyvoice", "CosyVoice-300M-25Hz"),
        ),
        "CosyVoice-300M-SFT": (
            "iic/CosyVoice-300M-SFT",
            os.path.join("data", "models", "cosyvoice", "CosyVoice-300M-SFT"),
        ),
        "CosyVoice-300M-Instruct": (
            "iic/CosyVoice-300M-Instruct",
            os.path.join("data", "models", "cosyvoice", "CosyVoice-300M-Instruct"),
        ),
        "CosyVoice-ttsfrd": (
            "iic/CosyVoice-ttsfrd",
            os.path.join("data", "models", "cosyvoice", "CosyVoice-ttsfrd"),
        ),
    }
    downloaded = []
    for model in selected_models:
        id, local_dir = model_mapping.get(model, (None, None))
        if id:
            snapshot_download(id, local_dir=local_dir)
            downloaded.append(model)
    return "Downloaded: " + ", ".join(downloaded)


def run_conda_install(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = (
            "Conda Install Output:\n" + result.stdout + "\nErrors:\n" + result.stderr
        )
    except Exception as e:
        output = f"Error: {e}"
    return output


def run_pip_install(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = "Pip Install Output:\n" + result.stdout + "\nErrors:\n" + result.stderr
    except Exception as e:
        output = f"Error: {e}"
    return output


def ui_core():
    gr.Markdown(
        """
    # CosyVoice2 TTS

    CosyVoice2 is a TTS solution for fluent and natural speech synthesis, supporting:
    - Zero-shot TTS (voice cloning)
    - Instruction-based TTS
    - Cross-lingual TTS with fine-grained control
    - Chunked text processing for long inputs

    The model supports English and Chinese.
    """
    )

    try:
        import WeTextProcessing  # noqa: F401

    except ImportError:
        gr.Markdown(
            """
            CosyVoice2 requires WeTextProcessing for text processing.
            Try installing with the install dependencies tab or manually:
    
    Install pynini with conda:
    ```bash
    conda install -c conda-forge pynini==2.1.6
    ```
    Then
    ```bash
    pip install WeTextProcessing
    ```
    """
        )

    with gr.Tabs():
        with gr.Tab("Zero-Shot TTS"):
            ui_app_tts()
        with gr.Tab("Instruct TTS"):
            ui_app_instruct()
        with gr.Tab("Cross-Lingual"):
            ui_app_cross_lingual()
        with gr.Tab("Chunked Text"):
            ui_app_chunk()
        with gr.Tab("Download Models"):
            gr.Markdown("## Download Pretrained Models")
            models_checkbox = gr.CheckboxGroup(
                choices=[
                    "CosyVoice2-0.5B",
                    "CosyVoice-300M",
                    "CosyVoice-300M-25Hz",
                    "CosyVoice-300M-SFT",
                    "CosyVoice-300M-Instruct",
                    "CosyVoice-ttsfrd",
                ],
                label="Select Models to Download",
            )
            download_btn = gr.Button("Download Selected Models", variant="primary")
            download_output = gr.Textbox(label="Download Status", interactive=False)
            download_btn.click(
                fn=download_models, inputs=models_checkbox, outputs=download_output
            )
        with gr.Tab("Install Dependencies"):
            gr.Markdown("## Install Dependencies")
            with gr.Row():
                conda_cmd_input = gr.Textbox(
                    value="conda install -c conda-forge modelscope",
                    label="Conda Install Command",
                    lines=1,
                )
                run_conda_btn = gr.Button("Run Conda Install", variant="primary")
                conda_output = gr.Textbox(label="Conda Output", interactive=False)
                run_conda_btn.click(
                    fn=run_conda_install, inputs=conda_cmd_input, outputs=conda_output
                )
            with gr.Row():
                pip_cmd_input = gr.Textbox(
                    value="pip install modelscope", label="Pip Install Command", lines=1
                )
                run_pip_btn = gr.Button("Run Pip Install", variant="primary")
                pip_output = gr.Textbox(label="Pip Output", interactive=False)
                run_pip_btn.click(
                    fn=run_pip_install, inputs=pip_cmd_input, outputs=pip_output
                )


def ui_app():
    with gr.Blocks() as app:
        ui_core()
    return app
