# CrewAI Sandbox

This is a playground repo where I test out [CrewAI](https://crewai.com), a framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks.

## Directory Structure

```text
CrewAISandbox/
├── Agent/                # Jupyter notebooks exploring basic agent creation and interaction
├── astro_crewai/         # A standard CrewAI crew project set up with UV for dependency management
├── astro_flow/           # Experimental workflows integrating CrewAI with different flow setups
├── default_crewai/       # CrewAI crew template with agents and tasks configured for research and reporting
└── default_flow/         # Additional experimental flow configurations for multi-agent systems
```

## Setup Instructions

### 1. Prerequisites

Make sure you have Python 3.10 or higher installed. This project uses `uv` to handle dependencies, which you'll need to install if you haven't already:

```bash
pip install uv
```

### 2. Installation

Navigate to the specific CrewAI project directory you want to run and install its dependencies. For example:

```bash
cd astro_crewai
crewai install
```

### 3. Environment Variables

Create a `.env` file in the specific project directory (e.g., `astro_crewai/.env`) and add your API keys. Since you're using Gemini:

```env
GEMINI_API_KEY="your-google-gemini-api-key"
# Add other keys depending on the models you use:
# OPENAI_API_KEY="your-openai-api-key"
```

### 4. Running a Crew

To kickstart your crew of AI agents and begin task execution, run this from the project folder:

```bash
crewai run
```

## Contributing

This is mainly a personal sandbox project, but feel free to open a PR or issue if you have suggestions or find bugs, or if there's an interesting pattern you'd like to share.
