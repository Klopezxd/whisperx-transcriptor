#!/usr/bin/env python3
"""
Aplicación Web Gradio para WhisperX Transcriptor.
Diseñada para despliegue en Hugging Face Spaces (ZeroGPU) y ejecución local.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import gradio as gr

from src.config import ModelName, TimestampMode, TranscriptionConfig
from src.transcriber import WhisperXPipeline
from src.utils import TranscriptionError, clear_gpu_vram, setup_logger

logger = setup_logger("whisperx_app")

# Compatibilidad con ZeroGPU en Hugging Face Spaces
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(func):
        return func


@GPU_DECORATOR
def process_transcription(
    media_file,
    model_name: str,
    enable_diarization: bool,
    timestamp_option: str,
    custom_token: str,
    progress=gr.Progress()
) -> tuple[str, str | None]:
    """Procesa el archivo multimedia y retorna el texto y la ruta del archivo generado."""
    if not media_file:
        return "❌ Por favor selecciona un archivo de audio o video.", None

    media_path = media_file.name if hasattr(media_file, "name") else str(media_file)

    # Mapeo de formato de timestamps
    mode_map = {
        "Simple ([HH:MM:SS])": TimestampMode.SIMPLE,
        "Rango ([Inicio - Fin])": TimestampMode.RANGE,
        "Ninguno (Texto limpio)": TimestampMode.NONE
    }
    selected_mode = mode_map.get(timestamp_option, TimestampMode.SIMPLE)

    token_to_use = custom_token.strip() if custom_token and custom_token.strip() else os.getenv("HF_TOKEN")

    if enable_diarization and not token_to_use:
        return (
            "❌ Para activar la identificación de hablantes (diarización), "
            "debes proporcionar un Token de Hugging Face o configurarlo en los Secrets del Space.",
            None
        )

    def on_progress(p: float, msg: str) -> None:
        progress(p, desc=msg)

    try:
        config = TranscriptionConfig.create_default(
            model_name=model_name,
            enable_diarization=enable_diarization,
            hf_token=token_to_use
        )
        object.__setattr__(config, "timestamp_mode", selected_mode)

        pipeline = WhisperXPipeline(config)
        result = pipeline.process(
            media_path=media_path,
            progress_callback=on_progress
        )

        return result.formatted_text, str(result.output_path)

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return f"❌ Error: {str(e)}", None
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en app web")
        return f"❌ Ocurrió un fallo inesperado: {str(e)}", None


# Construcción de la Interfaz Gráfica Gradio
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, title="WhisperX Transcriptor Pro") as demo:
    gr.Markdown(
        """
        # 🎙️ WhisperX Transcriptor Pro
        **Transcripción de alta fidelidad, alineación fonética de palabras y diarización de hablantes.**
        *Desarrollado con WhisperX, PyTorch y Pyannote. Compatible con ZeroGPU.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_media = gr.File(
                label="📁 Archivo Multimedia (MP4, MP3, WAV, MKV, M4A)",
                file_types=["video", "audio"]
            )

            with gr.Accordion("⚙️ Opciones de Configuración", open=True):
                model_selector = gr.Dropdown(
                    choices=[
                        ModelName.LARGE_V3_TURBO.value,
                        ModelName.LARGE_V3.value,
                        ModelName.MEDIUM.value,
                        ModelName.SMALL.value,
                        ModelName.BASE.value
                    ],
                    value=ModelName.LARGE_V3_TURBO.value,
                    label="🤖 Modelo Whisper",
                    info="'large-v3-turbo' ofrece el mejor equilibrio entre velocidad y máxima precisión."
                )

                timestamp_selector = gr.Radio(
                    choices=[
                        "Simple ([HH:MM:SS])",
                        "Rango ([Inicio - Fin])",
                        "Ninguno (Texto limpio)"
                    ],
                    value="Simple ([HH:MM:SS])",
                    label="⏰ Formato de Marcas de Tiempo"
                )

                diarization_checkbox = gr.Checkbox(
                    value=False,
                    label="👥 Identificar Hablantes (Diarización)",
                    info="Activa la separación de personas usando Pyannote Audio (requiere Token de Hugging Face)."
                )

                token_input = gr.Textbox(
                    label="🔑 Hugging Face Token (Opcional si está en variables de entorno)",
                    placeholder="hf_...",
                    type="password"
                )

            btn_transcribe = gr.Button("🚀 Iniciar Transcripción", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="📄 Vista Previa de la Transcripción",
                placeholder="El texto transcrito aparecerá aquí...",
                lines=18
            )
            download_file = gr.File(label="💾 Descargar Archivo (.txt)")

    btn_transcribe.click(
        fn=process_transcription,
        inputs=[
            input_media,
            model_selector,
            diarization_checkbox,
            timestamp_selector,
            token_input
        ],
        outputs=[output_text, download_file]
    )

    gr.Markdown(
        """
        ---
        💡 **Acerca de este proyecto:** Desarrollado por [Klever López](https://github.com/Klopezxd) | 
        Código fuente disponible en [GitHub](https://github.com/Klopezxd/whisperx-transcriptor)
        """
    )

if __name__ == "__main__":
    demo.launch()
