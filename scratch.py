from src.ai.analyst import build_llm, _text

llm = build_llm()
print(_text(llm.invoke("Reply with the single word: OK")))