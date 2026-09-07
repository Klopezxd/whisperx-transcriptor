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
):
    """Procesa el archivo multimedia y retorna el texto y el componente de descarga."""
    if not media_file:
        return (
            "⚠️ Por favor selecciona o arrastra un archivo de audio o video.",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )

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
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
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

        return (
            result.formatted_text,
            gr.DownloadButton(
                label=f"Descargar {result.output_path.name}",
                value=str(result.output_path),
                interactive=True,
            ),
        )

    except TranscriptionError as e:
        clear_gpu_vram()
        logger.error("Error en procesamiento web: %s", e)
        return (
            f"❌ Error de transcripción: {e}",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )
    except Exception as e:
        clear_gpu_vram()
        logger.exception("Error inesperado en app web")
        return (
            f"❌ Ocurrió un error inesperado durante el procesamiento: {e}",
            gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        )


# Estilos CSS y Tema visual refinado (adaptable a escritorio y móvil)
custom_css = """
.gradio-container {
    max-width: 1400px !important;
    width: 96% !important;
    margin: 0 auto !important;
}
#app-header {
    text-align: center;
    padding: 1.25rem 0 0.75rem 0;
}
#app-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
#app-header p {
    color: #64748b;
    font-size: 1rem;
}
.btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}
#footer-note {
    text-align: center;
    margin-top: 1.5rem;
    font-size: 0.85rem;
    color: #94a3b8;
}

/* Responsividad para móviles y pantallas compactas */
@media (max-width: 768px) {
    .gradio-container {
        width: 100% !important;
        padding: 0.5rem !important;
    }
    #app-header h1 {
        font-size: 1.55rem;
    }
    #app-header p {
        font-size: 0.9rem;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

# Compatibilidad entre Gradio < 6.0 y Gradio >= 6.0 (theme y css pasaron de Blocks a launch)
blocks_kwargs = {"title": "WhisperX Transcriptor"}
launch_kwargs = {}

try:
    _gradio_major = int(gr.__version__.split(".")[0])
    if _gradio_major >= 6:
        launch_kwargs["theme"] = theme
        launch_kwargs["css"] = custom_css
    else:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = custom_css
except Exception:
    blocks_kwargs["theme"] = theme
    blocks_kwargs["css"] = custom_css

with gr.Blocks(**blocks_kwargs) as demo:
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
        with gr.Column(scale=1):
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
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Resultado de la Transcripción",
                placeholder="El texto transcrito aparecerá aquí...",
                lines=14,
            )
            download_file = gr.DownloadButton(
                label="Descargar Documento (.txt)",
                value=None,
                interactive=False,
                size="lg",
                variant="secondary",
            )

    # Conexión de eventos
    btn_clear.add([input_media, output_text])
    btn_clear.click(
        fn=lambda: gr.DownloadButton(label="Descargar Documento (.txt)", value=None, interactive=False),
        outputs=download_file,
    )

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
    demo.launch(**launch_kwargs)

