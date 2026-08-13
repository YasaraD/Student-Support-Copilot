# AGENTS.md

## Project Identity

Project name:

**Student Support Copilot: A RAG-Based University Assistance Chatbot**

Repository name:

`student-support-copilot`

This project is a university student-support chatbot that uses Retrieval-Augmented Generation, or RAG, to answer questions using approved university documents.

The first version covers four knowledge categories:

1. Examinations
2. Modules
3. Student Services
4. Academic Regulations

Read `PROJECT_SCOPE.md` before making significant architectural or implementation decisions.

---

## Developer Context

The project owner is a Computer Science undergraduate building a RAG system for the first time.

Use beginner-friendly explanations throughout development.

When introducing a new technology, library, architectural pattern, or technical term:

1. Explain what it is in simple language.
2. Explain why it is needed in this project.
3. Explain where it fits into the RAG workflow.
4. Explain its main input and output.
5. Mention a simpler alternative when one exists.
6. Do not assume previous experience with RAG systems.
7. Include a small example when it improves understanding.

Examples of technologies and concepts that must be explained before they are introduced include:

* LangChain
* LangGraph
* Qdrant
* Chroma
* FAISS
* embedding models
* embeddings
* vector databases
* vector stores
* semantic search
* retrievers
* rerankers
* text chunking
* chunk size
* chunk overlap
* prompt templates
* conversation memory
* tool calling
* agent tools
* Docker
* APIs
* environment variables

Avoid introducing advanced technologies before they are required by the current milestone.

---

## Selected RAG Framework

LangChain has been selected as the primary framework for implementing the RAG pipeline.

Do not use LlamaIndex in this project unless the project owner explicitly requests a future comparison or migration.

LangChain must be introduced incrementally. Do not install or implement every LangChain component at once.

The planned LangChain implementation order is:

1. LangChain document loading
2. LangChain text splitting
3. Embedding-model integration
4. Vector-store integration
5. Retriever creation
6. Prompt-template creation
7. LLM integration
8. Complete RAG pipeline
9. Conversation-aware question handling
10. Evaluation and optimization

For each LangChain component introduced:

1. explain the underlying RAG concept;
2. explain what the LangChain component does;
3. explain why it is required for the current milestone;
4. show where it fits in the project workflow;
5. explain its input and output;
6. keep the implementation visible and beginner-friendly.

Do not hide the entire RAG workflow inside one large chain or function.

Keep separate modules for:

* document loading;
* text splitting;
* embeddings;
* vector storage;
* retrieval;
* prompt construction;
* answer generation;
* source formatting.

Use LangChain `Document` objects as the standard format for passing text and metadata through the RAG pipeline.

A LangChain `Document` should conceptually contain:

```python
Document(
    page_content="The extracted university document text",
    metadata={
        "source": "document_name.pdf",
        "filename": "document_name.pdf",
        "page": 1,
        "category": "examinations"
    }
)
```

Preserve useful metadata whenever possible, including:

* source filename;
* document title;
* category;
* page number;
* section heading;
* document version;
* publication or effective date.

LangGraph must not be introduced until the basic RAG chatbot has been implemented, tested, and evaluated.

Do not implement agents during the initial RAG milestones.

---

## Development Approach

Build the project incrementally.

Do not attempt to implement the complete RAG system in one task.

Use the following milestone order.

### Milestone 1: Project Setup and Streamlit Interface

* Create the project structure.
* Create the Streamlit chatbot interface.
* Add chat input and message display.
* Add temporary session history.
* Add a clear-conversation button.
* Display the four supported categories.
* Use a safe placeholder chatbot response.

Status:

**Completed**

### Milestone 2: Document Loading with LangChain

* Load one text-based PDF.
* Use LangChain’s `PyPDFLoader`.
* Return LangChain `Document` objects.
* Preserve page-level metadata.
* Add document inspection to Streamlit.
* Handle invalid, unreadable, scanned, or empty PDFs.
* Do not use OCR yet.

Status:

**Current milestone**

### Milestone 3: Text Splitting with LangChain

* Use a LangChain text splitter.
* Begin with `RecursiveCharacterTextSplitter`.
* Split long documents into retrieval-friendly chunks.
* Preserve the original document metadata.
* Add chunk identifiers.
* Inspect chunk size and overlap.
* Display sample chunks for development testing.

Status:

**Not started**

### Milestone 4: Embeddings

* Select one embedding model.
* Explain what embeddings are.
* Convert document chunks into numerical vectors.
* Test the embedding model with sample text.
* Keep the embedding provider configurable.

Status:

**Not started**

### Milestone 5: Local Vector Store

* Select one local vector store.
* Compare suitable options before choosing.
* Store document embeddings.
* Preserve metadata.
* Save the vector index locally where appropriate.
* Avoid Qdrant unless persistent or hosted storage becomes necessary.

Status:

**Not started**

### Milestone 6: Retrieval

* Convert the vector store into a LangChain retriever.
* Retrieve relevant chunks for student questions.
* Display the raw retrieved chunks before connecting an LLM.
* Test retrieval across the four categories.
* Add similarity thresholds or metadata filters only when justified.

Status:

**Not started**

### Milestone 7: LLM Integration

* Select one LLM provider.
* Store the API key using environment variables.
* Build a prompt that uses retrieved document evidence.
* Instruct the LLM not to invent missing information.
* Generate a student-friendly answer.
* Do not allow the LLM to make official university decisions.

Status:

**Not started**

### Milestone 8: Complete RAG Pipeline

Connect:

```text
Student question
        ↓
Retriever
        ↓
Relevant document chunks
        ↓
Prompt template
        ↓
LLM
        ↓
Answer with supporting sources
```

The chatbot must display:

* the generated answer;
* the supporting source filename;
* the relevant page number when available;
* the category;
* an uncertainty message when evidence is insufficient.

Status:

**Not started**

### Milestone 9: Add the Four Knowledge Categories

Add approved documents for:

* examinations;
* modules;
* student services;
* academic regulations.

Test questions from each category.

Status:

**Not started**

### Milestone 10: Evaluation

Evaluate:

* retrieval accuracy;
* source correctness;
* answer faithfulness;
* answer relevance;
* unsupported-question handling;
* response time;
* category coverage;
* user satisfaction.

Status:

**Not started**

### Milestone 11: Conversation-Aware RAG

After the basic RAG system works:

* support follow-up questions;
* rewrite context-dependent questions when necessary;
* avoid allowing conversation history to replace official document evidence;
* clearly separate chat history from retrieved knowledge.

Status:

**Not started**

### Milestone 12: Agents and LangGraph

Consider agent features only after the core RAG chatbot has been successfully implemented and evaluated.

Possible future agents include:

* support-ticket creation;
* appointment booking;
* satisfaction surveys;
* FAQ-gap identification;
* staff escalation.

Status:

**Future extension**

Complete and validate one milestone before moving to the next milestone.

Do not begin the next milestone unless explicitly requested.

---

## Initial Technology Constraints

Use:

* Python as the main programming language;
* Streamlit for the user interface;
* LangChain as the RAG framework;
* a Python virtual environment;
* environment variables for secrets;
* Git for version control.

For the initial prototype:

* prefer a simple local setup;
* prefer the smallest reasonable number of dependencies;
* use only the LangChain packages required by the current milestone;
* do not use LangGraph;
* do not implement AI agents;
* do not use Docker;
* do not add authentication;
* do not add cloud deployment;
* do not connect to a real university database;
* do not access private student records;
* do not introduce Qdrant until a persistent or hosted vector database is genuinely required;
* do not implement OCR unless scanned-document support becomes an approved milestone;
* do not add multiple vector stores;
* do not add multiple embedding providers;
* do not add multiple LLM providers during the initial implementation.

Do not use both LangChain and LlamaIndex.

When a new package is needed, add only the specific LangChain integration package required for the current milestone.

Examples may include:

* `langchain-community` for selected community integrations;
* `langchain-text-splitters` for text splitting;
* a provider-specific LangChain package for embeddings or LLM access;
* one vector-store integration.

Do not install unrelated LangChain packages in advance.

Use current, supported LangChain APIs.

Avoid deprecated imports, classes, methods, and chain patterns.

---

## Supported Knowledge Categories

The knowledge base will contain documents from the following categories.

### Examinations

Examples include:

* examination regulations;
* repeat examination procedures;
* deferred examination procedures;
* examination registration;
* medical certificate requirements;
* result appeals;
* examination misconduct;
* examination timetables;
* examination eligibility.

### Modules

Examples include:

* module descriptions;
* module credits;
* prerequisites;
* module registration;
* assessment weightings;
* module progression;
* repeating modules;
* core and optional modules;
* module learning outcomes.

### Student Services

Examples include:

* library services;
* IT support;
* career guidance;
* internship support;
* counselling information;
* student letters;
* student support offices;
* accessibility services;
* student complaint procedures.

### Academic Regulations

Examples include:

* attendance requirements;
* late submissions;
* extensions;
* academic appeals;
* plagiarism;
* academic misconduct;
* mitigating circumstances;
* progression rules;
* suspension of studies;
* withdrawal procedures.

---

## RAG Knowledge-Base Rules

The RAG knowledge base must contain university information documents, not project-management documents.

Do not add the following files to the RAG vector store:

* `AGENTS.md`;
* `PROJECT_SCOPE.md`;
* `README.md`;
* source-code documentation;
* developer notes;
* test files;
* environment files;
* application logs.

These files explain how to build or operate the application. They are not trusted sources for answering student questions.

RAG knowledge documents should be placed under category-specific folders inside:

`documents/raw/`

Use this structure:

```text
documents/
└── raw/
    ├── examinations/
    ├── modules/
    ├── student_services/
    └── academic_regulations/
```

Processed files, generated chunks, indexes, or other derived data may be stored under:

```text
documents/processed/
```

or:

```text
data/
```

Do not edit or overwrite original source documents during processing.

Preserve useful document metadata whenever possible, including:

* document title;
* category;
* filename;
* page number;
* section heading;
* document version;
* publication date;
* effective date.

Never place confidential student information in the knowledge base.

Do not use documents containing:

* student names;
* registration numbers;
* examination results;
* medical information;
* financial details;
* account credentials;
* private correspondence.

Only use documents that are:

* public;
* fictional;
* created specifically for testing;
* or explicitly approved for this project.

---

## Document-Loading Rules

Use LangChain’s document-loader interfaces.

For the initial PDF milestone:

* use `PyPDFLoader`;
* use `pypdf` as the PDF-reading dependency;
* load PDFs page by page;
* return LangChain `Document` objects;
* normalize metadata consistently.

Every loaded page should contain at least:

```python
{
    "source": "document_name.pdf",
    "filename": "document_name.pdf",
    "page": 1,
    "category": "examinations"
}
```

Accepted category identifiers are:

```text
examinations
modules
student_services
academic_regulations
```

Handle the following cases clearly:

* file does not exist;
* file is not a PDF;
* invalid category;
* PDF cannot be opened;
* encrypted PDF;
* unsupported PDF;
* empty PDF;
* PDF with no extractable text;
* scanned PDF without machine-readable text.

Do not silently return an empty list when extraction fails.

Do not add OCR during the initial document-loading milestone.

Uploaded PDFs must not be permanently saved by default.

When temporary files are required:

* use secure temporary-file handling;
* delete temporary files after processing;
* delete temporary files when errors occur;
* do not use hard-coded absolute paths.

---

## Text-Splitting Rules

When Milestone 3 begins, use LangChain’s text-splitting package.

Begin with:

`RecursiveCharacterTextSplitter`

Before selecting chunk settings, explain:

* what a chunk is;
* why a full document should not be embedded as one item;
* what chunk size means;
* what chunk overlap means;
* why overly small or overly large chunks can reduce retrieval quality.

Preserve the original metadata in every generated chunk.

Add useful chunk metadata such as:

* `chunk_id`;
* original page number;
* original filename;
* category.

Do not choose chunk size and overlap without documenting the reason.

Do not permanently fix chunk values without allowing later experimentation.

---

## Embedding Rules

When the embedding milestone begins:

* select one embedding model;
* explain what an embedding is;
* explain that the embedding model is different from the answer-generating LLM;
* explain how semantic similarity works;
* keep the embedding model configurable;
* do not mix vectors from different embedding models in the same index;
* record the embedding model name used to create an index.

Do not regenerate embeddings unnecessarily on every Streamlit rerun.

Use Streamlit caching or another controlled persistence approach only after explaining its purpose.

---

## Vector-Store Rules

Select only one vector store for the first working version.

Before selecting it:

1. compare the minimum suitable options;
2. explain whether the store is in-memory or persistent;
3. explain how it integrates with LangChain;
4. explain where vectors and metadata will be stored;
5. explain how the index can be rebuilt.

Do not introduce Qdrant unless the project requires:

* persistent hosted storage;
* multi-user access;
* more advanced filtering;
* production-style deployment;
* or a larger dataset.

Do not add multiple vector databases for experimentation during the initial version.

## Selected Vector Store

The initial vector store for the project is:

`Chroma`

Use the dedicated LangChain integration:

`Chroma` from `langchain_chroma`.

The vector store must run locally with persistent on-disk storage.

Use:

`data/chroma_db/`

as the default persistence directory.

The initial collection name is:

`student_support_knowledge`

Use the existing `BAAI/bge-m3` LangChain embedding model returned by the project's embedding module.

Do not create a second embedding-model configuration inside the vector-store module.

The vector store must contain:

* the dense BGE-M3 embedding;
* original chunk text;
* chunk ID;
* source filename;
* page number;
* category;
* existing useful metadata.

Use each chunk's deterministic `chunk_id` as its vector-store document ID where supported.

Configure the initial Chroma collection to use cosine distance because the project uses normalized text embeddings and evaluates semantic similarity using cosine-style comparison.

Use the current supported Chroma/LangChain configuration API. Do not use deprecated configuration parameters simply because they appear in older examples.

The initial implementation must use local embedded Chroma only.

Do not use:

* Chroma Cloud;
* a separate Chroma server;
* Docker;
* FAISS;
* Qdrant;
* another vector database.

Provide explicit operations for:

1. creating or opening the local vector store;
2. indexing chunked LangChain Documents;
3. checking stored document count;
4. rebuilding the development vector store when source documents or embedding configuration change;
5. reopening the persisted store without re-embedding all documents.

Do not silently append duplicate chunks every time Streamlit reruns.

Index construction must not happen automatically during every Streamlit rerun.

Do not implement the retriever or student-question similarity search until the retrieval milestone.

---

## Retrieval Rules

Before connecting the LLM, retrieval must work independently.

For a sample question, display:

* the retrieved text chunks;
* source filenames;
* page numbers;
* categories;
* relevance scores when available.

Verify that the correct document sections are being retrieved.

Do not allow an LLM to hide retrieval failures.

Keep retrieval settings configurable, including:

* number of returned chunks;
* metadata filters;
* similarity threshold when supported.

Do not assume that retrieving more chunks always improves the answer.

## Selected Retrieval Strategy

The initial retrieval strategy is dense semantic similarity retrieval using:

- BAAI/bge-m3 query embeddings;
- the existing persistent Chroma vector store;
- cosine-based vector search;
- LangChain retrieval interfaces.

The initial retrieval configuration should use:

TOP_K = 4

Treat this value as an experimental starting point rather than an optimal setting.

Student queries must use the same BAAI/bge-m3 embedding model used when indexing document chunks.

Retrieval must be validated independently before connecting an LLM.

The initial retrieval milestone must support:

1. unfiltered similarity retrieval across the complete knowledge base;
2. optional metadata filtering by one of the four project categories;
3. returning original LangChain Document content and metadata;
4. displaying retrieval scores or distances for development inspection when supported;
5. preserving source, filename, page, category, and chunk_id;
6. clear handling of an empty query;
7. clear handling of a missing or empty vector store.

Do not implement automatic category classification during the initial retrieval milestone.

Do not implement query rewriting, reranking, MMR, hybrid retrieval, sparse retrieval, multi-query retrieval, or LLM-based retrieval during the initial retrieval milestone.

Start with straightforward dense similarity search.

Keep retrieval logic in `src/retriever.py`.

Do not place retrieval logic directly inside `app.py`.

Do not connect an LLM until retrieval has been manually and programmatically evaluated.
---

## LLM Rules

The LLM is the language-generation and reasoning component of the system.

It is not the knowledge base.

The LLM must receive:

* the student’s question;
* relevant retrieved document chunks;
* clear system instructions;
* source metadata where needed.

The LLM must be instructed to:

* answer using the retrieved evidence;
* avoid inventing policies;
* state when the evidence is insufficient;
* provide a clear student-friendly explanation;
* avoid making official university decisions;
* distinguish guidance from formal approval.

Do not connect the LLM before retrieval has been tested independently.

Never hard-code the model API key.

Keep the model name configurable through environment variables or a configuration module.

## Selected LLM Architecture

The initial answer-generation configuration is:

```text
Provider: Google Gemini Developer API
Model: gemini-3.5-flash-lite
LangChain package: langchain-google-genai
LangChain interface: ChatGoogleGenerativeAI
Architecture: 2-Step RAG
```

Use the existing BAAI/bge-m3 retriever, persistent Chroma store, and
`TOP_K = 4`. Gemini is used only for answer generation, not embeddings or
retrieval. Do not use Google File Search or send complete source PDFs to Gemini;
send only the current student question and retrieved text chunks needed for that
question.

Do not introduce another LLM provider, conversation-aware RAG, or agents during
this milestone.

---

## Answering and Retrieval Requirements

When the RAG pipeline is implemented, it must:

* answer using retrieved document evidence;
* display the supporting document source;
* display page numbers when available;
* avoid inventing policies or procedures;
* clearly state when sufficient information cannot be found;
* distinguish informational guidance from official university decisions;
* recommend contacting the appropriate university department when formal confirmation is required;
* avoid presenting general LLM knowledge as official university policy.

A suitable insufficient-evidence response is:

> I could not find enough information in the available university documents to answer this question confidently. Please contact the relevant university department for official confirmation.

The chatbot must never claim that it can:

* approve an appeal;
* approve an extension;
* change examination registrations;
* register a module;
* access private marks;
* make an official academic decision;
* approve mitigating circumstances;
* issue official university documents.

---

## Streamlit Interface Rules

Streamlit is responsible for the user interface.

Keep Streamlit responsibilities separate from RAG-processing responsibilities.

`app.py` should manage:

* page configuration;
* interface layout;
* chat input;
* chat messages;
* session state;
* source display;
* user-facing errors;
* development inspection panels.

The `src/` modules should manage:

* document loading;
* text splitting;
* embeddings;
* vector storage;
* retrieval;
* prompts;
* LLM calls;
* RAG orchestration.

Do not place the entire RAG pipeline directly inside `app.py`.

Development-only tools such as document previews or chunk inspection should be placed inside a clearly labelled Streamlit expander or sidebar section.

Do not display an entire uploaded document by default.

Do not expose internal error traces to ordinary users.

---

## Code Quality Rules

Write beginner-friendly and maintainable Python.

Code should:

* use clear names;
* use small, focused functions;
* include type hints where practical;
* include concise docstrings for important functions;
* avoid unnecessary abstraction;
* avoid duplicated logic;
* handle expected errors clearly;
* keep user-interface logic separate from RAG-processing logic;
* avoid placing the entire application in one large file;
* use constants for repeated configuration values;
* use consistent metadata names;
* avoid deeply nested logic;
* avoid unexplained magic numbers.

Do not leave unexplained placeholder code in completed milestones.

Do not silently ignore errors.

When an error occurs, provide a useful message explaining:

* what failed;
* the likely cause;
* what the user can do next.

Do not use broad exception handling such as:

```python
except Exception:
    pass
```

When a broad exception must be caught at a user-interface boundary, log or preserve enough information for development debugging while showing a safe user-facing message.

---

## Project Structure Guidance

Use a simple structure that can grow gradually:

```text
student-support-copilot/
├── AGENTS.md
├── PROJECT_SCOPE.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── app.py
│
├── documents/
│   ├── raw/
│   │   ├── examinations/
│   │   ├── modules/
│   │   ├── student_services/
│   │   └── academic_regulations/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   └── document_loader.py
│
├── data/
│   └── .gitkeep
│
└── tests/
    ├── __init__.py
    └── test_document_loader.py
```

Do not create every possible RAG module before it is needed.

Add modules only during their relevant milestones.

Possible future modules include:

```text
src/
├── document_loader.py
├── text_splitter.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── prompts.py
├── llm.py
├── source_formatter.py
└── rag_pipeline.py
```

Do not create empty future modules solely to make the structure appear complete.

---

## Secrets and Security

Never hard-code:

* API keys;
* access tokens;
* passwords;
* private database credentials.

Use environment variables.

Provide a `.env.example` file containing placeholder variable names only.

Example:

```text
LLM_API_KEY=
LLM_MODEL=
EMBEDDING_MODEL=
```

The real `.env` file must be excluded through `.gitignore`.

Do not print secrets to:

* the terminal;
* application logs;
* test output;
* the Streamlit interface;
* exception messages.

Do not commit uploaded documents unless they are public, fictional, or explicitly approved for the project.

Do not send documents to an external AI provider during milestones that only require local document loading, chunking, or inspection.

---

## Dependency Rules

Before adding a dependency:

1. Explain what the dependency does.
2. Explain why the current milestone requires it.
3. Check whether an existing dependency already provides the same function.
4. Add it to `requirements.txt`.
5. Avoid unnecessary packages.
6. Avoid installing multiple libraries that perform the same role.
7. Use package versions that are compatible with the existing project.

Use only the LangChain packages required for the current milestone.

For the document-loading milestone, the expected new dependencies are:

```text
langchain-community
pypdf
```

Do not add an LLM provider package during document loading.

Do not add a vector-store package during document loading.

Do not install LangGraph until the agent milestone.

Do not add dependencies only because they may be useful later.

After modifying dependencies:

* install them in the project virtual environment;
* verify that imports work;
* update `requirements.txt`;
* report what was added and why.

---

## Testing and Validation

After changing code:

1. Check that imports work.
2. Check Python syntax.
3. Run available tests.
4. Check that the Streamlit application starts.
5. Report all validation failures.
6. Fix errors caused by the current milestone.
7. Do not claim that something works unless it was validated.

For document loading, test at least:

* valid category handling;
* invalid category rejection;
* missing-file handling;
* non-PDF rejection;
* successful metadata normalization;
* no-extractable-text handling.

Mocking external or file-processing components is acceptable for focused unit tests.

When a suitable small PDF is available, also perform one manual end-to-end document-loading validation.

Do not add large test documents to the repository unnecessarily.

For each completed task, report:

* files created;
* files modified;
* dependencies added;
* commands run;
* tests run;
* validation results;
* remaining limitations;
* the recommended next milestone.

---

## Documentation Rules

Update `README.md` after each completed milestone.

The README should explain:

* what the project does;
* the current milestone;
* what has been completed;
* what has not been implemented;
* how to create and activate the virtual environment;
* how to install dependencies;
* how to start Streamlit;
* how to run tests;
* where knowledge documents should be placed;
* current limitations.

Do not describe the project as a complete RAG chatbot until retrieval and LLM generation are connected.

For every introduced LangChain component, add a short beginner-friendly explanation to the README.

---

## Change Management

Before making major architectural changes:

1. explain the proposed change;
2. explain why it is needed;
3. identify new dependencies;
4. identify affected files;
5. keep the change limited to the current milestone.

Do not refactor unrelated parts of the project during a focused task.

Do not delete user-created files without explicit permission.

Do not overwrite `PROJECT_SCOPE.md` unless specifically requested.

Do not modify the agreed four project categories without explicit permission.

Do not replace LangChain with another RAG framework without explicit permission.

Do not begin a future milestone unless explicitly requested.

---

## Communication Style

Use clear, beginner-friendly explanations.

When completing a task:

* summarize what was implemented;
* explain important code sections;
* explain how the component fits into the RAG pipeline;
* explain how to run and test the result;
* mention what has not been implemented yet;
* report errors honestly;
* recommend only one logical next milestone.

Avoid giving only a list of code changes without explaining their purpose.

Do not describe the project as complete while planned RAG features are still missing.
