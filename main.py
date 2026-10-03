import asyncio
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, LLMConfig, Provider

# Modelo que se usa si OPENAI_MODEL / ANTHROPIC_MODEL están vacías o no existen
MODELOS_POR_DEFECTO = {
    Provider.OPENAI: "gpt-6-luna",
    Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
}

PREGUNTA = "¿Qué es la entropía?"


def cargar_config() -> LLMConfig:
    """Lee el .env y arma una configuración validada por Pydantic."""
    load_dotenv()

    provider = Provider(os.getenv("LLM_PROVIDER", "").strip().lower())

    variable_modelo = "OPENAI_MODEL" if provider == Provider.OPENAI else "ANTHROPIC_MODEL"
    # `or` cubre tanto la variable ausente como la variable vacía (OPENAI_MODEL=)
    model = os.getenv(variable_modelo) or MODELOS_POR_DEFECTO[provider]

    return LLMConfig(
        provider=provider,
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
    )


async def main() -> None:
    try:
        config = cargar_config()
        manager = AsyncLLMManager(config)
    except (ValueError, ValidationError) as e:
        print(f"Error de configuración: {e}")
        print("Revisá tu archivo .env (ver .env.example).")
        return

    print(f"Proveedor: {config.provider.value} | Modelo: {config.model}")
    mensajes = [ChatMessage(role="user", content=PREGUNTA)]

    print(f"\n=== Modo normal ===\n> {PREGUNTA}\n")
    respuesta = await manager.generate(mensajes)
    if respuesta.error:
        print(f"[Error] {respuesta.error}")
    else:
        print(respuesta.content)

    print(f"\n=== Modo streaming ===\n> {PREGUNTA}\n")
    async for fragmento in manager.generate_stream(mensajes):
        print(fragmento, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
