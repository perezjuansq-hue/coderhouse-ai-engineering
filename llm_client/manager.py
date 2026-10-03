from typing import AsyncGenerator, List

from llm_client.providers.anthropic_client import AnthropicClient
from llm_client.providers.base import BaseLLMClient
from llm_client.providers.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, Provider


class AsyncLLMManager:
    """Punto de entrada único: elige el cliente según la configuración y delega en él."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: BaseLLMClient = self._crear_cliente()

    def _crear_cliente(self) -> BaseLLMClient:
        """Fábrica: crea el cliente que corresponde al proveedor configurado."""
        if self.config.provider == Provider.OPENAI:
            if not self.config.openai_api_key:
                raise ValueError("Falta OPENAI_API_KEY en la configuración")
            return OpenAIClient(
                api_key=self.config.openai_api_key.get_secret_value(),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        if self.config.provider == Provider.ANTHROPIC:
            if not self.config.anthropic_api_key:
                raise ValueError("Falta ANTHROPIC_API_KEY en la configuración")
            return AnthropicClient(
                api_key=self.config.anthropic_api_key.get_secret_value(),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        raise ValueError(f"Proveedor no soportado: {self.config.provider}")

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Respuesta completa (modo normal)."""
        return await self._client.generate(messages)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Respuesta fragmento a fragmento (modo streaming)."""
        async for fragmento in self._client.generate_stream(messages):
            yield fragmento
