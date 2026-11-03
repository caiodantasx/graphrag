"""Cliente LLM mínimo para APIs compatíveis com OpenAI (chat/completions).

Funciona com OpenAI, Groq, OpenRouter, Ollama (``http://localhost:11434/v1``),
LM Studio e outros. Sem LLM configurado (``LLM_PROVIDER=none``), os módulos
caem para estratégias determinísticas (regras), mantendo o sistema executável.
"""
from __future__ import annotations

import json
import re

from .config import Settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float | None = None) -> str:
        if not self.enabled:
            raise LLMError("LLM desabilitado (LLM_PROVIDER=none)")
        import requests

        payload = {
            "model": self.s.llm_model,
            "temperature": self.s.llm_temperature if temperature is None else temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.s.llm_api_key:
            headers["Authorization"] = f"Bearer {self.s.llm_api_key}"
        try:
            r = requests.post(
                f"{self.s.llm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120,
            )
            if r.status_code == 400 and json_mode:
                # alguns servidores não suportam response_format
                payload.pop("response_format", None)
                r = requests.post(
                    f"{self.s.llm_base_url.rstrip('/')}/chat/completions",
                    json=payload, headers=headers, timeout=120,
                )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover - depende de rede
            raise LLMError(str(e)) from e

    def chat_json(self, system: str, user: str) -> dict:
        txt = self.chat(system, user, json_mode=True)
        return parse_json_loose(txt)


def parse_json_loose(txt: str) -> dict:
    """Extrai o primeiro objeto JSON de uma resposta (tolerante a ```json ...```)."""
    txt = txt.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if m:
        txt = m.group(1)
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"Resposta sem JSON: {txt[:200]}")
    return json.loads(txt[start : end + 1])
