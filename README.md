# deepReSearch

‼️WARNING‼️: This project is currently in early stage

## Get Started

### Installation

```shell
uv sync --dev
uv run crawl4ai-setup
```

### Run

```shell
uv run streamlit run demo.py
```

## Reflection during development

- We may need Receptionist, to directly respond greetings

- Simple question or task should be allowed to bypass the standard lengthy procedure

- Should consider topic derivation in early procedure?

- Is an useless document content really useless? Maybe LLM currently have not enough knowledge to judge

- Should LLM explain why a document is useless?

- Noticed that the word "Note" MAY downgrade the output style and limit the divergent thinking of LLM
