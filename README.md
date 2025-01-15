# Smithers

An LLM-powered CLI that simulates job interviews based on your resume.

## Example

```bash
pipx run smithers-llm [RESUME_PATH] --role=[ROLE]
```

## Installation

With pip:

```bash
pip install -g smithers-llm
```

With pipx (automatically adds smithers to PATH):

```bash
pipx install smithers-llm
```
 
> [!WARNING]
> Please notice that the user is expected to run the model separately as well.

So far the only model supported is Llama3.1

You can easily self-host it with [Ollama](https://ollama.com). After installing it, pull the model and keep it running on the background with the command below:

```bash
ollama run llama3.1
```

## Usage

With smithers installed on the current env:

```bash
python -m smithers-llm [RESUME_PATH] --role=[ROLE]
```

With smithers installed globally and defined on PATH:

```bash
smithers-llm [RESUME_PATH] --role=[ROLE]
```

With pipx:

```bash
pipx run smithers-llm [RESUME_PATH] --role=[ROLE]
```

## Learnings

- LangChain
  - Structured Outputs
- LangGraph
  - Human in the Loop
  - Checkpointer
- Python packaging with SetupTools and Build
  - Differences between project configurations files such as the modern pyproject.toml and the battle-tested setup.py and setup.cfg
- Publishing to PyPI and TestPyPI with Twine
- Creating CLIs with Click
  - Handling different kinds of options: parameters and arguments
- Experimenting with Jupyter Notebooks
  - Research whether it was worth commiting the notebooks, although they might be reasonable for documentation purposes, they weren't for the way I used them in this project, to scratch the general ideas up right in the beginning
- Self-reflection if RAG was necessary here, the answer I came to was NO. The files are too small.

## Fun fact

The repo was named after a character from the show "The Simpsons".

Smithers conducts a job interview with Homer on the episode ["I Married Marge"](https://www.youtube.com/watch?v=rG6w0IAoT4U).