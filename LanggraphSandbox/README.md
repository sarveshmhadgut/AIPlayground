# Langgraph Sandbox

This is a playground repository where I test out [LangGraph](https://langchain-ai.github.io/langgraph/), LLMs (mainly Gemini), and RAG (Retrieval-Augmented Generation) setups. It serves as a progressive exploration of LangGraph's capabilities, from basic workflows to advanced agentic patterns.

## Directory Structure

The repository is structured as a series of progressive explorations:

```text
LanggraphSandbox/
├── 01_sequential_workflows/    # Basic sequential agent flows
├── 02_parallel_workflows/      # Running tasks in parallel
├── 03_conditional_workflows/   # Branching logic and conditional edges
├── 04_iterative_workflows/     # Loops and cyclic graphs
├── 05_goofy_chatbot/           # A simple chatbot implementation
├── 06_persistence/             # State persistence and checkpoints
├── 07_streaming/               # Streaming LLM outputs
├── 08_threading/               # Multi-threading capabilities
├── 09_sqlite/                  # Checkpointing with SQLite
├── 10_observability/           # Tracing and observability
├── 11_tools/                   # Equipping agents with custom tools
├── 12_rag/                     # Retrieval-Augmented Generation setups
├── 13_hitl/                    # Human-in-the-loop (HITL) interactions
├── 14_stm/                     # Short-Term Memory implementations
├── 15_ltm/                     # Long-Term Memory implementations
├── Flaude/                     # LangGraph based Agentic Chatbot
└── langgraph_utils/            # Shared utilities for the sandbox
```

## Setup Instructions

### 1. Prerequisites

Make sure you have Python 3.11 or higher installed. This project uses `uv` for dependency management:

```bash
pip install uv
```

### 2. Installation

Clone the repository and install dependencies using `uv`:

```bash
git clone
cd AIPlayground/LanggraphSandbox

uv sync
```

### 3. Environment Variables

Create a `.env` file in the `LanggraphSandbox` directory and add your API keys:

```env
GOOGLE_API_KEY="your-google-gemini-api-key"
# Add other keys depending on the tools or models you use:
# OPENAI_API_KEY="your-openai-api-key"
# ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 4. Running the Code

Each directory represents a self-contained topic. You can run the Python scripts inside them using `uv run`. For example:

```bash
uv run python 01_sequential_workflows/main.py
```
*(Check inside each directory for the specific scripts available.)*

## Contributing

This is mainly a personal sandbox project, but feel free to open a PR or issue if you have suggestions or find bugs, or if there's an interesting pattern you'd like to share.
