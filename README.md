# Smithers

An LLM-powered CLI that simulates job interviews based on your resume.

## Installation

With pip:

```bash
pip install -g smithers
```

With pipx:

```bash
pipx install smithers
```
 
Please notice that so far the only model supported is Llama3.1

You can easily self-host it with [Ollama](https://ollama.com). After installing it, pull the model and keep it running on the background with the command below:

```bash
ollama run llama3.1
```

## Usage

```bash
smithers [RESUME_PATH] --role=[ROLE]
```

## Fun fact

The repo was named after a character from the show "The Simpsons".

Smithers conducts a job interview with Homer on the episode ["I Married Marge"](https://www.youtube.com/watch?v=rG6w0IAoT4U).