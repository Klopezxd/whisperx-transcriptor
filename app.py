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

# Compatibilidad con ZeroGPU en Hugging Face Spaces (hasta 120s de GPU dedicada)
try:
    import spaces

    GPU_DECORATOR = spaces.GPU(duration=120)
except (ImportError, TypeError, AttributeError):
    def GPU_DECORATOR(func):
        return func


@GPU_DECORATOR
def process_transcription(
    media_file,
    model_name: str,
    enable_diarization: bool,
    timestamp_option: str,
    progress=gr.Progress(),
) -> tuple[str, str | None]:
    """Procesa el archivo multimedia y retorna el texto y la ruta del archivo generado."""
    if not media_file:
        return "⚠️ Por favor selecciona o arrastra un archivo de audio o video.", None

    media_path = media_file.name if hasattr(media_file, "name") else str(media_file)

    # Mapeo de formato de marcas de tiempo
    mode_map = {
        "Simple ([HH:MM:SS])": TimestampMode.SIMPLE,
        "Rango ([Inicio - Fin])": TimestampMode.RANGE,
        "Texto continuo (Sin marcas)": TimestampMode.NONE,
    }
    selected_mode = mode_map.get(timestamp_option, TimestampMode.SIMPLE)

    token_to_use = os.getenv("HF_TOKEN")

    if enable_diarization and not token_to_use:
        return (
            "⚠️ La identificación de hablantes (diarización) requiere configurar 'HF_TOKEN' "
            "en las variables de entorno o Secrets del servidor.\n\n"
            "Consejo: Desmarca la casilla de diarización para transcribir directamente con Whisper.",
            None,
        )

    def on_progress(p: float, msg: str) -> None:
        progress(p, desc=msg)

    try:
        config = TranscriptionConfig.create_default(
            model_name=model_name,
            enable_diarization=enable_diarization,
            hf_token=token_to_use,
        )
        object.__setattr__(config, "timestamp_mode", selected_mode)

        pipeline = WhisperXPipeline(config)
        result = pipeline.process(
            media_path=media_path,
            progress_callback=on_progress,
        )

        return result.formatted_text, str(result.output_path)

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return f"❌ Error de transcripción: {e}", None
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en app web")
        return f"❌ Ocurrió un error inesperado durante el procesamiento: {e}", None


# Estilos CSS y Tema visual refinado
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
#app-header {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}
#app-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
#app-header p {
    color: #64748b;
    font-size: 1.05rem;
}
.btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}
#footer-note {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.88rem;
    color: #94a3b8;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme, css=custom_css, title="WhisperX Transcriptor") as demo:
    gr.HTML(
        """
        <div id="app-header">
            <h1>🎙️ WhisperX Transcriptor</h1>
            <p>Transcripción rápida de audio y video con alineación fonética y separación de interlocutores</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # Columna Izquierda: Entrada y Controles
        with gr.Column(scale=5):
            input_media = gr.File(
                label="Archivo Multimedia",
                file_types=["audio", "video"],
                file_count="single",
            )

            diarization_checkbox = gr.Checkbox(
                value=False,
                label="Identificar quién habla (Diarización)",
                info="Separa y etiqueta las intervenciones de cada locutor automáticamente.",
            )

            with gr.Accordion("Opciones avanzadas", open=False):
                model_selector = gr.Dropdown(
                    choices=[
                        ModelName.LARGE_V3_TURBO.value,
                        ModelName.LARGE_V3.value,
                        ModelName.MEDIUM.value,
                        ModelName.SMALL.value,
                        ModelName.BASE.value,
                    ],
                    value=ModelName.LARGE_V3_TURBO.value,
                    label="Modelo de Whisper",
                    info="'large-v3-turbo' ofrece máxima precisión con la velocidad más alta.",
                )

                timestamp_selector = gr.Radio(
                    choices=[
                        "Simple ([HH:MM:SS])",
                        "Rango ([Inicio - Fin])",
                        "Texto continuo (Sin marcas)",
                    ],
                    value="Simple ([HH:MM:SS])",
                    label="Marcas de tiempo",
                )

            with gr.Row():
                btn_clear = gr.ClearButton(
                    value="Limpiar",
                    variant="secondary",
                )
                btn_transcribe = gr.Button(
                    "Iniciar Transcripción",
                    variant="primary",
                    size="lg",
                    elem_classes=["btn-primary"],
                )

        # Columna Derecha: Salida y Descarga
        with gr.Column(scale=6):
            output_text = gr.Textbox(
                label="Resultado de la Transcripción",
                placeholder="El texto transcrito aparecerá aquí...",
                lines=16,
                show_copy_button=True,
            )
            download_file = gr.File(
                label="Descargar Documento (.txt)",
                interactive=False,
            )

    # Conexión de eventos
    btn_clear.add([input_media, output_text, download_file])

    btn_transcribe.click(
        fn=process_transcription,
        inputs=[
            input_media,
            model_selector,
            diarization_checkbox,
            timestamp_selector,
        ],
        outputs=[output_text, download_file],
    )

    gr.HTML(
        """
        <div id="footer-note">
            Desarrollado con <strong>WhisperX</strong>, <strong>Wav2Vec2</strong> y <strong>Pyannote Audio</strong> · Acelerado con <strong>ZeroGPU</strong><br>
            <a href="https://github.com/Klopezxd/whisperx-transcriptor" target="_blank" style="color: inherit; text-decoration: underline;">Código fuente en GitHub</a>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch()
