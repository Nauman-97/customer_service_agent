# Customer Service Agent

An AI-powered, knowledge-grounded email support agent for small businesses. It classifies incoming customer emails, retrieves relevant information from a local business knowledge base, drafts a response, and records conversation history for more consistent follow-up.

The project is designed as a Python application that can be integrated into an existing email or messaging workflow. The repository currently provides the agent logic and a JSON-based batch runner; connecting a live email provider or messaging platform requires an external adapter.

## Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Add Your Business Knowledge](#add-your-business-knowledge)
- [Run the Agent](#run-the-agent)
- [Integrate the Agent Into Your Application](#integrate-the-agent-into-your-application)
- [Input and Output](#input-and-output)
- [Data, Logs, and Metrics](#data-logs-and-metrics)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap and Integration Boundaries](#roadmap-and-integration-boundaries)
- [License](#license)

## Overview

Customer Service Agent uses an LLM-based classification and response pipeline with retrieval-augmented generation (RAG). Business documents are indexed in a persistent ChromaDB vector store. For each email, the agent summarizes the inquiry for retrieval, searches for relevant knowledge, considers recent conversation history, and produces a structured draft response.

The agent returns classification metadata alongside the proposed reply, including category, priority, spam probability, and flags indicating whether the case should be escalated or closed. These fields can help a downstream support workflow route the message or request human review.

## Features

- **Email classification:** Categorizes messages as `Order Status`, `Refund Request`, `Complaint`, `Product Question`, `Others`, or `Spam`.
- **Priority assessment:** Assigns `Critical`, `High`, `Medium`, or `Low` priority.
- **Spam scoring:** Produces a spam probability between 0 and 1.
- **Knowledge-grounded replies:** Retrieves relevant business information using semantic similarity search before generating a response.
- **Multiple document formats:** Supports PDF, DOCX, TXT, Markdown, HTML, and HTM knowledge files.
- **Persistent vector search:** Stores embeddings and indexed document chunks in ChromaDB.
- **Incremental knowledge-base synchronization:** Tracks file and chunk hashes to avoid reprocessing unchanged content; checks for updates on a configured interval.
- **Conversation context:** Stores previous email and response details in SQLite and uses recent history for classification and response generation.
- **Structured LLM outputs:** Uses Pydantic schemas to validate classification and response fields.
- **Retry handling:** Retries LLM calls and database operations where configured, with validation feedback for structured model output.
- **Usage and performance metrics:** Records estimated token usage, processing time, and model/embedding cost estimates in a JSON metrics file.
- **Logging:** Writes component logs to a local `logs/` directory.
- **Python integration:** Exposes a `CustomerAgent` class for integration into other Python applications.
- **Batch processing:** Includes a JSON-file runner for processing multiple emails.

## How It Works

```text
Incoming email
     |
     v
Validate input (Pydantic)
     |
     v
Classify email + inspect recent history
(category, priority, spam probability, summary)
     |
     v
Embed summary and search business knowledge (ChromaDB)
     |
     v
Generate structured reply using retrieved context + history
     |
     v
Validate result and save conversation history (SQLite)
     |
     v
Return reply, classification metadata, and escalation/closure flags
```

The primary stages are:

1. **Validate:** Check that the email contains `email_address`, `subject`, and `body`.
2. **Classify:** Use an OpenAI model to identify the message category, priority, spam probability, and a retrieval-oriented summary.
3. **Retrieve:** Embed the summary with `text-embedding-3-small` and retrieve up to three relevant knowledge chunks from ChromaDB.
4. **Respond:** Generate a structured reply using the retrieved context and recent messages for the same email address.
5. **Persist:** Store the response and related metadata in SQLite for future context.
6. **Report:** Return a success/error result and update local metrics and logs.

## Technology Stack

| Component | Technology |
| --- | --- |
| Language | Python |
| LLM classification and response generation | OpenAI Responses API |
| Embeddings | OpenAI `text-embedding-3-small` |
| Structured validation | Pydantic |
| Retrieval/vector storage | ChromaDB |
| Conversation history | SQLite |
| Document extraction | pypdf, python-docx, Beautiful Soup |
| Text processing | NLTK, tiktoken, NumPy |
| Configuration | python-dotenv |

The model identifiers and settings are configured in `customer_agent.py`. Model availability and pricing can change; confirm current access and pricing in your OpenAI account before running the agent.

## Project Structure

```text
customer_service_agent/
├── customer_agent.py     # Main CustomerAgent pipeline and metrics
├── main.py               # JSON batch-processing entry point
├── openai_client.py      # OpenAI API and embedding wrappers
├── prompts.py            # Classification and response instructions
├── schemas.py            # Input and structured-output Pydantic schemas
├── semantic_search.py    # Knowledge ingestion, chunking, embeddings, retrieval
├── email_history.py      # SQLite conversation-history persistence
├── utils.py              # Logging utilities
├── requirements.txt      # Python dependencies
├── README.md
├── knowledge_base/       # Add business documents here (created as needed)
├── data/
│   ├── vector_database/  # Persistent ChromaDB data (created at runtime)
│   └── email_history_database/
│       └── email_history.db
├── logs/                 # Component log files (created at runtime)
├── metrics.json          # Runtime usage/performance estimates
├── input_email.json      # Batch input file used by main.py
└── output.json           # Batch output file produced by main.py
```

Some runtime files and directories are generated automatically on first run. Do not commit API keys, customer messages, local databases, logs, or other sensitive runtime data to a public repository.

## Requirements

- Python version compatible with the pinned dependencies in `requirements.txt` (Python 3.11+ is a reasonable starting point; verify package compatibility for your environment).
- An OpenAI API key with access to the configured text-generation and embedding models.
- Internet access for OpenAI API requests and, on first startup, any NLTK tokenizer data that is not already installed.
- Business knowledge documents to ground responses. The knowledge base may be empty at startup, but retrieval-based replies require indexed content to be useful.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nauman-97/customer_service_agent.git
cd customer_service_agent
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the API key

Create a file named `.env` in the repository root:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
```

Replace the placeholder with your own key. Keep `.env` private and out of version control. Do not hard-code API keys in source files or share them in logs, screenshots, or issue reports.

## Configuration

The main model and pipeline settings are currently defined as class attributes in `customer_agent.py`, rather than being exposed as environment variables.

| Setting | Current behavior |
| --- | --- |
| Classifier model | `gpt-5-mini-2025-08-07` |
| Responder model | `gpt-5.4-mini-2026-03-17` |
| Embedding model | `text-embedding-3-small` |
| Retrieved knowledge chunks | Up to 3 |
| LLM retries | 5 configured attempts, with a 2-second delay |
| History retries | 5 configured attempts, with a 0.2-second delay |
| Knowledge sync interval | 600 seconds |
| Maximum retrieved context | 12,000 characters |

If you change model identifiers, make sure the models are available to your account and compatible with the API calls and structured-output schemas used by the project. The cost figures in the code are estimates and should be reviewed against current pricing.

## Add Your Business Knowledge

1. Create or use the `knowledge_base/` directory in the project root.
2. Add relevant business materials in PDF, DOCX, TXT, Markdown, HTML, or HTM format.
3. Include accurate product information, shipping policies, returns/refunds policies, FAQs, contact details, and other information the agent is allowed to use.
4. Run the application. The semantic-search component synchronizes the directory and stores document embeddings in the persistent ChromaDB collection.
5. Test representative customer questions and review the generated replies for accuracy and completeness.

The retriever returns up to three relevant chunks, with a maximum combined context size of 12,000 characters. Keep source documents current, remove obsolete policies, and test important questions after updating the knowledge base. Retrieval improves grounding but does not guarantee that every answer will be correct.

## Run the Agent

### Process emails from a JSON file

The included `main.py` reads `input_email.json` from the current working directory and writes results to `output.json`.

Create `input_email.json`:

```json
[
  {
    "email_address": "customer@example.com",
    "subject": "Question about my order",
    "body": "Hello, could you please tell me how long shipping usually takes?"
  },
  {
    "email_address": "another.customer@example.com",
    "subject": "Return policy",
    "body": "What is your return policy for an unused item?"
  }
]
```

Run:

```bash
python main.py
```

The runner processes the supplied emails sequentially and writes the returned results to `output.json`. The agent may retry unsuccessful calls; failed processing after the configured retry limit can raise an exception. Review the logs if a batch stops unexpectedly.

### Use the agent directly in Python

```python
from customer_agent import CustomerAgent

agent = CustomerAgent()

email = {
    "email_address": "customer@example.com",
    "subject": "Shipping question",
    "body": "How long does standard shipping take?"
}

result = agent.reply(email)

if result[0]["agent_success"]:
    response = result[1]
    print(response["reply"])
    print("Category:", response["category"])
    print("Priority:", response["priority"])
    print("Escalate:", response["escalate"])
else:
    print("Agent failed:", result[0])
```

For a convenience method that raises an exception if processing continues to fail, use `agent.get_response(email, retries=3)`. See the return formats below before integrating either method.

## Integrate the Agent Into Your Application

The `CustomerAgent` class can be called from a backend service, support dashboard, scheduled job, or an adapter for an email/messaging provider.

A typical integration flow is:

1. Receive a message through your provider's webhook, API, or polling mechanism.
2. Map the provider payload to the required input fields: `email_address`, `subject`, and `body`.
3. Call `CustomerAgent.reply(email)` or `CustomerAgent.get_response(email, retries=3)`.
4. Inspect the result and handle errors explicitly.
5. Use the returned `reply` as a **draft** or send it through your provider only after your application’s approval policy allows it.
6. Route messages according to `category`, `priority`, `spam_probability`, and `escalate`.
7. If your provider supports message IDs, delivery status, or idempotency keys, manage them in your integration layer to prevent duplicate processing and sending.

**Important:** This repository does not currently implement a Gmail/Outlook connector, inbound webhook server, SMTP sender, WhatsApp/Slack integration, or deployment service. Those integrations must be built separately using the relevant provider's API and authentication flow. The agent generates the response content; it does not itself deliver email to the customer.

### Example integration boundary

```python
def handle_incoming_message(provider_message: dict, agent):
    email = {
        "email_address": provider_message["email_address"],
        "subject": provider_message["subject"],
        "body": provider_message["body"],
    }

    result = agent.reply(email)

    if not result[0]["agent_success"]:
        # Send to your application's retry or error-handling queue.
        return {"success": False, "error": result[0]}

    response = result[1]

    if response["escalate"]:
        # Route to a human support queue in your integration.
        return {"success": True, "action": "escalate", "draft": response}

    # Store or display the draft; send only according to your approval policy.
    return {"success": True, "action": "review", "draft": response}
```

This is an integration pattern, not a built-in provider connector. Adapt field mapping and error handling to the source platform.

## Input and Output

### Required input

Each email must be a dictionary containing these string fields:

| Field | Description |
| --- | --- |
| `email_address` | Customer's email address; also used to look up conversation history |
| `subject` | Email subject |
| `body` | Email body |

### Successful response

`CustomerAgent.reply(email)` returns a two-item list:

- First item: status dictionary, for example `{"agent_success": true}`.
- Second item: structured response dictionary when successful.

The response dictionary includes:

| Field | Description |
| --- | --- |
| `email_address` | Customer email address |
| `subject` | Original subject |
| `body` | Original email body |
| `summary` | Concise summary used for retrieval |
| `category` | Classified inquiry category |
| `priority` | Assigned priority |
| `spam_probability` | Estimated spam probability from 0 to 1 |
| `reply` | Draft response generated by the model |
| `escalate` | Whether human intervention is recommended |
| `close` | Whether the case is marked as requiring no further reply |

### Failure response

On failure, `reply()` returns a one-item list containing a status dictionary with `agent_success: false`, a `failure` stage, and error details. Validate the success flag before accessing the response payload.

`get_response(email, retries=3)` returns the response dictionary on success and raises `RuntimeError` if processing still fails after its retry loop. Note that `retries` refers to additional attempts after an initial call in the current implementation.

## Data, Logs, and Metrics

The application creates and maintains local runtime data:

- **Conversation history:** `data/email_history_database/email_history.db` (SQLite, using write-ahead logging).
- **Vector database:** `data/vector_database/` (persistent ChromaDB collection named `knowledge_vectors`).
- **Knowledge-base indexes:** `knowledge_base/.kb_index.json` and `knowledge_base/.chunk_index.json`.
- **Logs:** `logs/customer_agent.log`, `logs/openai_client.log`, `logs/email_history.log`, and `logs/semantic_search.log`.
- **Metrics:** `metrics.json`, containing aggregate counts, processing-time averages, token estimates, and estimated costs for classification, retrieval, and response generation.

Metrics are local estimates derived from the implementation and configured pricing values; they are not a billing statement. Protect these files because logs, databases, and metrics may reveal customer or business information.

## Operational Notes

- **Human review:** Treat generated replies as drafts until you have evaluated accuracy, tone, and policy compliance. Keep human escalation available for refunds, complaints, sensitive cases, and other high-impact decisions.
- **Knowledge quality:** Retrieval can only use the information available in the knowledge base. Missing, stale, or contradictory documents can produce incomplete answers.
- **Privacy:** Send only data you are authorized to process to the external LLM API. Review your business's privacy obligations and the provider's data-handling terms before using real customer messages.
- **Security:** Store API keys in environment variables or a secret manager. Restrict access to local databases, logs, and knowledge files.
- **Reliability:** Add provider-level timeouts, idempotency, monitoring, and a durable retry/dead-letter queue when deploying in a live workflow.
- **Concurrency:** The current implementation uses local files and persistent databases. Evaluate synchronization and shared-state behavior before running multiple agent instances concurrently.
- **Cost control:** LLM and embedding calls may incur charges. Monitor actual usage and set appropriate limits in your OpenAI account.

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| Authentication or API errors | Confirm `OPENAI_API_KEY` is present, valid, and has access to the configured models. |
| Empty or irrelevant replies | Add relevant documents to `knowledge_base/`, confirm their text is extractable, and test with clear questions. |
| Knowledge files are not reflected immediately | The index synchronizes on startup and then uses a 600-second interval; restart the application or explicitly trigger a sync in your own code when appropriate. |
| Structured-output validation errors | Review `logs/openai_client.log` and confirm the configured model supports the required structured outputs. |
| Batch run fails | Check the input JSON syntax, required fields, dependency installation, API limits, and component logs. |
| Local database or file errors | Confirm the process has read/write permissions in the project directory and that local runtime files are accessible. |

## Roadmap and Integration Boundaries

Potential next steps for production use include:

- Provider-specific adapters for Gmail, Microsoft Outlook, or other support channels.
- A webhook/API layer and background job queue.
- Human approval workflows and a support-agent dashboard.
- Tests, CI, and end-to-end evaluation datasets for classification and answer quality.
- Configurable model names, retry behavior, and knowledge-base paths through environment variables.
- Monitoring, rate-limit handling, access controls, and deployment documentation.

These are possible extensions, not claims of existing functionality.

## License

This project includes a [MIT License](LICENSE).
