# LangChain Sandbox

This is a playground repo where I test out LangChain, LLMs (mainly Gemini), and RAG setups.

## Directory Structure

```text
LangchainSandbox/
├── Chains/               # Execution paths and complex chains
├── DocumentLoaders/      # Parsers for PDFs, text, and web
├── GoofyChatbot/         # Persona testing and memory handling
├── Misc/                 # Miscellaneous scripts and OpenAI tests
├── Models/               # Boilerplate for foundation models
├── OutputParser/         # Parsers for LLM responses (JSON, Pydantic, String, Structured)
├── PromptsNMessages/     # Prompt templates and chat orchestration
├── Retrievers/           # Finding data using Vector DBs
├── Runnables/            # Core LangChain Runnables (lambda, parallel, passthrough, sequential)
├── StructuredOutput/     # Structured output validations (Pydantic, TypedDict)
├── TLDRify/              # YouTube transcript summarizer and Streamlit app
├── TextSplitters/        # Chunking strategies for embeddings
├── ToolCalling/          # Tool calling / function calling examples
└── VectorStores/         # Local DB setups (Chroma, FAISS)
```

## Setup Instructions

### 1. Prerequisites

Make sure you have Python 3.11 or higher installed. This project uses `uv` to handle dependencies, which you'll need to install if you haven't already:

```bash
pip install uv
```

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone
cd AIPlayground/LangchainSandbox

uv sync
```

If you prefer `pip`, you can also do:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the `LangchainSandbox` directory and add your API keys.

```env
GOOGLE_API_KEY="your-google-gemini-api-key"
# Add other keys depending on the models you use:
# OPENAI_API_KEY="your-openai-api-key"
# ANTHROPIC_API_KEY="your-anthropic-api-key"
# HUGGINGFACE_API_KEY="your-hf-token"
```

### 4. Running the YouTube RAG App (TLDRify)

To spin up the interactive Streamlit app for YouTube RAG:

```bash
cd TLDRify
uv run streamlit run app.py
```

## Contributing

This is mainly a personal sandbox project, but feel free to open a PR or issue if you have suggestions or find bugs, or if there's an interesting pattern you'd like to share.
