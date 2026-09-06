"""
Configuración centralizada y modelos de datos para WhisperX Transcriptor.
Aplica principios de Clean Code con inmutabilidad y validación de tipos.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class ModelName(str, Enum):
    """Modelos Whisper soportados."""
    LARGE_V3_TURBO = "large-v3-turbo"
    LARGE_V3 = "large-v3"
    MEDIUM = "medium"
    SMALL = "small"
    BASE = "base"


class ComputeType(str, Enum):
    """Tipos de computación para cuantización de VRAM."""
    FLOAT16 = "float16"
    INT8 = "int8"
    FLOAT32 = "float32"


class TimestampMode(str, Enum):
    """Formatos de visualización para marcas de tiempo."""
    SIMPLE = "simple"       # [HH:MM:SS]
    RANGE = "range"         # [HH:MM:SS - HH:MM:SS]
    NONE = "none"           # Sin marcas de tiempo


@dataclass(frozen=True)
class TranscriptionConfig:
    """
    Configuración inmutable para la ejecución del pipeline de transcripción.

    Attributes:
        model_name: Nombre o identificador del modelo Whisper.
        device: Dispositivo de cómputo ('cuda' o 'cpu').
        compute_type: Precisión o cuantización ('float16', 'int8', 'float32').
        batch_size: Tamaño de lote para la inferencia.
        enable_diarization: Si se debe ejecutar la diarización de hablantes.
        timestamp_mode: Formato de las marcas de tiempo en la salida.
        hf_token: Token de Hugging Face para pyannote (requerido si diarization=True).
        language: Código de idioma ISO (opcional, detección automática por defecto).
    """
    model_name: str = ModelName.LARGE_V3_TURBO.value
    device: str = "cuda"
    compute_type: str = ComputeType.INT8.value
    batch_size: int = 16
    enable_diarization: bool = False
    timestamp_mode: TimestampMode = TimestampMode.SIMPLE
    hf_token: Optional[str] = None
    language: Optional[str] = None

    @classmethod
    def create_default(
        cls,
        model_name: str = ModelName.LARGE_V3_TURBO.value,
        enable_diarization: bool = False,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ) -> "TranscriptionConfig":
        """Crea una configuración con selección automática de hardware."""
        import torch

        detected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        detected_compute = compute_type or ("int8" if detected_device == "cuda" else "int8")

        return cls(
            model_name=model_name,
            device=detected_device,
            compute_type=detected_compute,
            enable_diarization=enable_diarization,
            hf_token=hf_token
        )
