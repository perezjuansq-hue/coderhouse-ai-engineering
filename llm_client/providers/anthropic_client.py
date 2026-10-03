from typing import AsyncGenerator, List
from anthropic import (
    AsyncAnthropic,
    omit,
    APIError as AnthropicAPIError,
    RateLimitError as AnthropicRateLimitError,
    APIConnectionError as AnthropicConnectionError,
)
from llm_client.providers.base import BaseLLMClient
from llm_client.schemas import ChatMessage, ModelResponse, Provider


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _separar_mensajes(self, messages: List[ChatMessage]) -> tuple[str | None, list[dict]]:
        """Anthropic no acepta role="system" dentro de `messages`: va en el parámetro `system`.

        Devuelve (texto de system o None, resto de los mensajes como diccionarios).
        """
        textos_system = [m.content for m in messages if m.role == "system"]
        conversacion = [m.model_dump() for m in messages if m.role != "system"]
        system = "\n\n".join(textos_system) if textos_system else None
        return system, conversacion

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            system, conversacion = self._separar_mensajes(messages)
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,  # obligatorio en Anthropic, a diferencia de OpenAI
                # SDK 1.x quitó `temperature` de la firma; la API la sigue aceptando en Haiku 4.5
                extra_body={"temperature": self.temperature},
                system=system or omit,  # `omit` = no enviar el parámetro si no hay system
                messages=conversacion,
            )
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content=response.content[0].text,
            )
        except AnthropicRateLimitError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Límite de cuota excedido: {e}")
        except AnthropicConnectionError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error de conexión: {e}")
        except AnthropicAPIError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error de la API de Anthropic: {e}")
        except Exception as e:  # red de seguridad: cualquier error imprevisto
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error inesperado: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            system, conversacion = self._separar_mensajes(messages)
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                # SDK 1.x quitó `temperature` de la firma; la API la sigue aceptando en Haiku 4.5
                extra_body={"temperature": self.temperature},
                system=system or omit,
                messages=conversacion,
            ) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"
        except Exception as e:  # red de seguridad: cualquier error imprevisto
            yield f"\n[⚠️ Error inesperado durante el streaming: {e}]"
