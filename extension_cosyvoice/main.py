import gradio as gr


def extension__tts_generation_webui():
    cosyvoice_ui()
    return {
        "package_name": "extension_cosyvoice",
        "name": "CosyVoice",
        "version": "0.1.6",
        "requirements": "git+https://github.com/nilreml/extension_cosyvoice@main",
        "description": "CosyVoice: A TTS solution for fluent and natural speech synthesis.",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "rsxdalv",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/rsxdalv/CosyVoice",
        "extension_website": "https://github.com/rsxdalv/extension_cosyvoice",
        "extension_platform_version": "0.0.1",
    }


def cosyvoice_ui():
    from extension_cosyvoice.gradio_app import ui_core

    ui_core()


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        extension__tts_generation_webui()
    demo.launch(
        server_port=7770,
    )
