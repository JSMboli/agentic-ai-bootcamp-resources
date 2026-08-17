# References and Further Reading

## Agent orchestration and human oversight

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph quickstart and tool loop: https://docs.langchain.com/oss/python/langgraph/quickstart
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph persistence/checkpointing: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph agentic RAG: https://docs.langchain.com/oss/python/langgraph/agentic-rag
- LangChain human-in-the-loop: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

## Groq

- Groq quickstart: https://console.groq.com/docs/quickstart
- Groq active/supported models: https://console.groq.com/docs/models
- Qwen 3.6 27B model page: https://console.groq.com/docs/model/qwen/qwen3.6-27b
- Groq API reference: https://console.groq.com/docs/api-reference
- Groq tool-use overview: https://console.groq.com/docs/tool-use/overview
- Groq local tool calling: https://console.groq.com/docs/tool-use/local-tool-calling
- Groq built-in tools: https://console.groq.com/docs/tool-use/built-in-tools
- LangChain ChatGroq integration: https://docs.langchain.com/oss/python/integrations/chat/groq

## Ollama and local models

- Ollama API introduction: https://docs.ollama.com/api/introduction
- Ollama chat endpoint: https://docs.ollama.com/api/chat
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling
- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- Ollama context length: https://docs.ollama.com/context-length
- Official Ollama Python library: https://github.com/ollama/ollama-python
- LangChain ChatOllama integration: https://docs.langchain.com/oss/python/integrations/chat/ollama

## Other provider extension routes

- OpenAI developer documentation: https://developers.openai.com/
- Gemini API documentation: https://ai.google.dev/gemini-api/docs
- LangChain Google Generative AI integration: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
- Anthropic developer documentation: https://docs.anthropic.com/
- DeepSeek API documentation: https://api-docs.deepseek.com/

## Suggested critical-reading questions for students

1. When is a fixed workflow safer and easier to evaluate than a model-driven agent loop?
2. What state must be checkpointed for a multi-step agent to resume safely?
3. Which tool calls should require human approval before execution rather than review afterwards?
4. How should an agent behave when retrieval returns weak, contradictory or missing evidence?
5. How would you distinguish a provider-connectivity failure from a model-capability failure?
6. What evidence would justify moving from a deterministic workflow to greater model autonomy?
7. What tests detect hallucination, incorrect tool choice, bad tool arguments, unsafe automation and stale evidence?
8. When does local inference improve privacy or cost, and what capability/maintenance trade-offs does it introduce?

## August 2026 provider/version checks

- Groq deprecations (check before teaching): https://console.groq.com/docs/deprecations
- Groq reasoning controls: https://console.groq.com/docs/reasoning
- OpenAI current model catalogue: https://developers.openai.com/api/docs/models
- OpenAI quickstart / Responses API: https://developers.openai.com/api/docs/quickstart
- Gemini current model catalogue: https://ai.google.dev/gemini-api/docs/models
- Gemini API-key guide: https://ai.google.dev/gemini-api/docs/api-key
- Anthropic current model overview: https://docs.anthropic.com/en/docs/about-claude/models/overview
- DeepSeek current models/pricing: https://api-docs.deepseek.com/quick_start/pricing
