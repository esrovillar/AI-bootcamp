"""
week2_ai_chat.py: la parada 5 del asistente de Café Aurora, por sí sola.

Funciona con LM Studio (local) u OpenAI (nube): USE_LOCAL_LLM en .env decide.
    python week2_ai_chat.py            haces una pregunta, recibes texto
    python week2_ai_chat.py extract    pegas una reseña, recibes ReviewData
"""

import os
import sys
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an assistant for the staff of Café Aurora, a coffee shop. "
    "Answer general questions accurately, in the language of the question, in at most three sentences. "
    "You have NO information about Café Aurora itself (menu, prices, hours, policies). "
    "If asked about it, say you do not have that information yet."
)

def load_config() -> dict[str, Any]:
    """Lee .env, lo valida y devuelve la configuración. Lanza ValueError."""
    load_dotenv()
    use_local = os.getenv("USE_LOCAL_LLM", "true").strip().lower() == "true"
    config: dict[str, Any] = {
        "use_local": use_local,
        "max_tokens": int(os.getenv("MAX_TOKENS", "150")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "timeout": float(os.getenv("TIMEOUT", "30")),
        "max_retries": int(os.getenv("MAX_RETRIES", "2")),
    }
    if use_local:
        config["base_url"] = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        config["api_key"] = "not-needed"
        config["model"] = os.getenv("MODEL", "llama-3.2-3b-instruct")
        config["provider"] = "LM Studio (local)"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "USE_LOCAL_LLM=false pero OPENAI_API_KEY está vacía.\n"
                "  1. Pídele a tu instructor la key del curso\n"
                "  2. Agrégala a .env: OPENAI_API_KEY=sk-proj-...\n"
                "  O pon USE_LOCAL_LLM=true para usar LM Studio (gratis)."
            )
        config["base_url"] = None  # None = el default del SDK, https://api.openai.com/v1
        config["api_key"] = api_key
        config["model"] = os.getenv("MODEL", "gpt-4o-mini")
        config["provider"] = "OpenAI (nube)"
    return config


def main() -> int:
    """Carga la config, verifica el servidor y corre el chat."""
    print("🚀 Week 2 AI Chat · Café Aurora")
    try:
        config = load_config()
    except ValueError as error:
        print(f"❌ Error de configuración:\n{error}")
        return 1
    print(f"✅ Proveedor: {config['provider']} · modelo: {config['model']}")
    client = create_client(config)
    if config["use_local"] and not check_server(client, config):
        return 1
    return run_chat(client, config)

def create_client(config: dict[str, Any]) -> OpenAI:
    """El mismo cliente para los dos proveedores: solo cambian base_url y api_key."""
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=config["timeout"],
        max_retries=config["max_retries"],
    )

def check_server(client: OpenAI, config: dict[str, Any]) -> bool:
    """Solo LM Studio: GET /v1/models antes de preguntar nada."""
    try:
        model_ids = [model.id for model in client.models.list()]
    except openai.APIConnectionError:
        print(f"❌ LM Studio no responde en {config['base_url']}")
        print("💡 Abre LM Studio, carga el modelo y enciende el servidor (pestaña Developer).")
        return False
    # Con varias variantes del mismo modelo, LM Studio agrega "@variante" al id.
    base_ids = {model_id.split("@")[0] for model_id in model_ids}
    if config["model"] not in model_ids and config["model"] not in base_ids:
        print(f"❌ El modelo '{config['model']}' no está disponible en LM Studio.")
        print(f"💡 Disponibles: {', '.join(model_ids) or '(ninguno)'}. Ajusta MODEL en .env.")
        return False
    print(f"✅ LM Studio está encendido con {config['model']}")
    return True

def call_llm(client: OpenAI, config: dict[str, Any], messages: list[dict[str, str]]) -> Any:
    """Una llamada al modelo. Devuelve la respuesta del SDK, o None tras explicar el error."""
    try:
        return client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
    except openai.APITimeoutError:
        print(f"❌ Sin respuesta tras {config['timeout']:g} s (y {config['max_retries']} reintentos).")
        print("💡 Local: cierra otras apps o usa un modelo más chico. Nube: intenta más tarde.")
    except openai.APIConnectionError:
        print(f"❌ No se pudo conectar con {config['provider']}.")
        print("💡 Local: ¿está encendido el servidor de LM Studio? Nube: revisa tu internet.")
    except openai.AuthenticationError:
        print("❌ 401: la API key fue rechazada. Reintentar no lo arregla.")
        print("💡 Copia de nuevo la key en .env (sin comillas ni espacios) o pídele otra a tu instructor.")
    except openai.RateLimitError as error:
        print(f"❌ 429 de {config['provider']}: {error.code or 'rate limit'}.")
        print("💡 Espera un minuto. Si el código es insufficient_quota, la key llegó a su límite: avísale a tu instructor.")
    except openai.APIStatusError as error:
        print(f"❌ HTTP {error.status_code} de {config['provider']}.")
        if error.status_code >= 500:
            print(f"💡 Problema del servidor; el SDK ya reintentó {config['max_retries']} veces. Intenta más tarde.")
        else:
            print(f"💡 {error.message}")
    return None

def run_chat(client: OpenAI, config: dict[str, Any]) -> int:
    """Modo chat: una pregunta, una respuesta."""
    question = input("💬 Tu pregunta: ").strip()
    if not question:
        print("❌ Pregunta vacía: no hay nada que enviar.")
        return 1
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    print(f"🔄 Llamando a {config['provider']}...")
    response = call_llm(client, config, messages)
    if response is None:
        return 1
    print(response.choices[0].message.content)
    return 0

if __name__ == "__main__":
    sys.exit(main())

    