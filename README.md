# coderhouse-ai-engineering

**Pre-entrega 1: Unified Async LLM Client.** Cliente asíncrono en Python 3.12 para hablar con **OpenAI** o **Anthropic** a través de una misma interfaz. El proveedor se elige desde el archivo `.env`, sin tocar el código.

## Características

- **Intercambiabilidad:** `OpenAIClient` y `AnthropicClient` heredan de una clase abstracta común (`BaseLLMClient`). El `AsyncLLMManager` crea el cliente que corresponde según la configuración.
- **Asincronía:** todas las llamadas usan los SDKs asíncronos oficiales (`AsyncOpenAI`, `AsyncAnthropic`) con `async`/`await`, así que no bloquean el event loop.
- **Streaming:** `generate_stream()` es un generador asíncrono que entrega el texto con `yield` a medida que llega de la API.
- **Validación con Pydantic:** los mensajes (`ChatMessage`), la configuración (`LLMConfig`: temperatura entre 0 y 2, `max_tokens` > 0, proveedor válido) y la respuesta (`ModelResponse`) están validados. Las API keys se guardan como `SecretStr`, que no se muestra al imprimir.
- **Errores controlados:** los errores de límite de tasa, conexión, API y cualquier error imprevisto se atrapan y se devuelven como un `ModelResponse` con el campo `error`, sin cortar el programa.

## Requisitos

- Python **3.12**
- Una API key de OpenAI y/o de Anthropic, según el proveedor que uses.

## Instalación

```bash
git clone https://github.com/perezjuansq-hue/coderhouse-ai-engineering.git
cd coderhouse-ai-engineering

# Crear el entorno virtual con Python 3.12
py -3.12 -m venv .venv          # Windows
# python3.12 -m venv .venv      # macOS / Linux

# Activarlo
source .venv/Scripts/activate   # Windows (Git Bash)
# .venv\Scripts\Activate.ps1    # Windows (PowerShell)
# source .venv/bin/activate     # macOS / Linux

# Instalar dependencias
python -m pip install -r requirements.txt
```

## Configuración

Copiá la plantilla y completá tus valores:

```bash
cp .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | Sí | Proveedor a usar: `openai` o `anthropic` (en minúsculas) |
| `OPENAI_API_KEY` | Si `LLM_PROVIDER=openai` | API key de OpenAI ([platform.openai.com](https://platform.openai.com/api-keys)) |
| `ANTHROPIC_API_KEY` | Si `LLM_PROVIDER=anthropic` | API key de Anthropic ([console.anthropic.com](https://console.anthropic.com/)). Tiene que pertenecer a un *workspace* |
| `OPENAI_MODEL` | No | Modelo de OpenAI. Por defecto: `gpt-6-luna` |
| `ANTHROPIC_MODEL` | No | Modelo de Anthropic. Por defecto: `claude-haiku-4-5-20251001` |

> El archivo `.env` está en `.gitignore` y **no se sube al repositorio**. Nunca subas tus API keys.

## Ejecución

```bash
python main.py
```

El script hace la pregunta *"¿Qué es la entropía?"* dos veces: primero en **modo normal**, donde muestra la respuesta completa al final, y después en **modo streaming**, donde el texto aparece a medida que se genera.

```
Proveedor: openai | Modelo: gpt-6-luna

=== Modo normal ===
> ¿Qué es la entropía?

La entropía es una medida de ...

=== Modo streaming ===
> ¿Qué es la entropía?

La entropía es una medida de ...
```

Para cambiar de proveedor, modificá `LLM_PROVIDER` en el `.env`. También podés hacerlo solo para una ejecución:

```bash
LLM_PROVIDER=anthropic python main.py    # Git Bash / macOS / Linux
```

Si la configuración es inválida (proveedor desconocido, falta la API key, etc.), el script muestra un mensaje de error claro y termina sin traceback.

## Uso como librería

```python
from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, LLMConfig

config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001",
                   anthropic_api_key="sk-ant-...", max_tokens=500)
manager = AsyncLLMManager(config)

mensajes = [
    ChatMessage(role="system", content="Respondé en una sola oración."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]

# Modo normal
respuesta = await manager.generate(mensajes)
print(respuesta.error or respuesta.content)

# Modo streaming
async for fragmento in manager.generate_stream(mensajes):
    print(fragmento, end="", flush=True)
```

## Estructura del proyecto

```
.
├── main.py                       # Script de prueba: modo normal + streaming
├── llm_client/
│   ├── schemas.py                # Modelos Pydantic: Provider, ChatMessage, LLMConfig, ModelResponse
│   ├── manager.py                # AsyncLLMManager: crea el cliente según el proveedor
│   └── providers/
│       ├── base.py               # BaseLLMClient: clase abstracta (contrato común)
│       ├── openai_client.py      # Implementación con AsyncOpenAI
│       └── anthropic_client.py   # Implementación con AsyncAnthropic
├── requirements.txt
├── .env.example                  # Plantilla de variables de entorno
└── README.md
```

## Decisiones de diseño

- **Mensajes `system`:** OpenAI los acepta dentro de `messages`, pero Anthropic los recibe en un parámetro aparte (`system`). `AnthropicClient` los separa automáticamente, así que el mismo `List[ChatMessage]` funciona con los dos proveedores.
- **Manejo de errores:** cada cliente atrapa primero los errores específicos del SDK (`RateLimitError`, `APIConnectionError`, `APIError`) y al final un `except Exception` como red de seguridad. En modo normal, el error vuelve en `ModelResponse.error`. En streaming, se entrega como un último fragmento de texto marcado con `[⚠️ ...]`.
- **Particularidades de los SDKs y modelos actuales:**
  - OpenAI: los modelos nuevos usan `max_completion_tokens` en lugar de `max_tokens`, y `gpt-6-luna` solo acepta `temperature=1`. Por eso la temperatura por defecto es `1.0`.
  - Anthropic: el SDK 1.x ya no acepta `temperature` como argumento, aunque la API sí la acepta en Haiku 4.5. Se envía con `extra_body`.
