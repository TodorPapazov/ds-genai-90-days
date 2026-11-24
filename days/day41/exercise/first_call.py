# Day 41, step 1: one chat call from Python. Roles, temperature, max_tokens, usage and cost.
#   python first_call.py          real API if LLM_API_KEY or XAI_API_KEY is set, otherwise the offline mock
#   python first_call.py --mock   always offline
import sys
import llm_client as lc

client = lc.get_client(force_mock="--mock" in sys.argv)
messages = [
    # system: the standing instructions (who the assistant is, rules, format)
    {"role": "system", "content": "You are a concise assistant for a hospital data analytics team. Answer in 2 sentences."},
    # user / assistant pairs: earlier turns. The API is stateless, so YOU resend the history every call.
    {"role": "user", "content": "What is a 30-day readmission?"},
    {"role": "assistant", "content": "A return inpatient stay within 30 days of discharge. Hospitals track it as a quality measure."},
    # the new question
    {"role": "user", "content": "Why would a care manager want follow-up dates pulled out of discharge notes?"},
]
resp = lc.chat_with_retries(client, messages, model="mock-extractor" if client.name == "mock" else lc.MODEL,
                            temperature=0.2, max_tokens=120, timeout=30)
choice = resp["choices"][0]
print(f"client: {client.name} | model: {resp.get('model')} | finish_reason: {choice['finish_reason']}")
print("reply:", choice["message"]["content"])
u = resp["usage"]
print(f"tokens: {u['prompt_tokens']} in + {u['completion_tokens']} out = {u['total_tokens']} | "
      f"cost about ${lc.cost_usd(u):.6f} | latency {resp['_latency_s']} s | retries {resp['_retries']}")
