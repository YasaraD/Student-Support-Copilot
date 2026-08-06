# AGENTS.md

## Project Identity

Project name:

**Student Support Copilot: A RAG-Based University Assistance Chatbot**

Repository name:

`student-support-copilot`

This project is a university student-support chatbot that will eventually use Retrieval-Augmented Generation, or RAG, to answer questions using approved university documents.

The first version covers four categories:

1. Examinations
2. Modules
3. Student Services
4. Academic Regulations

Read `PROJECT_SCOPE.md` before making significant architectural or implementation decisions.

---

## Developer Context

The project owner is a Computer Science undergraduate building a RAG system for the first time.

When introducing a new technology, library, architectural pattern, or technical term:

1. Explain what it is in simple language.
2. Explain why it is needed in this project.
3. Explain where it fits into the RAG workflow.
4. Mention a simpler alternative when one exists.
5. Do not assume prior experience with RAG systems.

Examples of technologies that must be explained before introduction include:

* LangChain
* LlamaIndex
* LangGraph
* Qdrant
* Chroma
* FAISS
* embedding models
* vector databases
* retrievers
* rerankers
* chunking strategies
* prompt templates
* conversation memory
* agent tools
* Docker

Avoid introducing advanced technologies before they are required by the current milestone.

---

## Development Approach

Build this project incrementally.

Do not attempt to implement the entire RAG system in one task.

Use the following development order:

1. Create the basic project structure.
2. Create the Streamlit chatbot interface.
3. Load one local document.
4. Extract text from the document.
5. Split the extracted text into chunks.
6. Generate embeddings.
7. Store embeddings in a local vector store.
8. Retrieve relevant document chunks.
9. Connect retrieval to a Generative AI model.
10. Display answers with sources.
11. Add the four document categories.
12. Add evaluation and testing.
13. Consider agents only after the core RAG chatbot works correctly.

Complete and validate one milestone before moving to the next milestone.

---

## Initial Technology Constraints

Use:

* Python as the main programming language
* Streamlit for the user interface
* a Python virtual environment
* environment variables for secrets
* Git for version control

For the initial prototype:

* prefer a simple local setup;
* prefer the smallest reasonable number of dependencies;
* do not use LangGraph;
* do not implement AI agents;
* do not use Docker;
* do not add authentication;
* do not add cloud deployment;
* do not connect to a real university database;
* do not access private student records;
* do not introduce Qdrant until a persistent or hosted vector database is genuinely required.

Do not select both LangChain and LlamaIndex. When a RAG framework becomes necessary, compare them briefly and recommend only one for this project.

Do not add a framework when plain Python can clearly handle the current milestone.

---

## Supported Knowledge Categories

The knowledge base will contain documents from these categories:

### Examinations

Examples include:

* examination regulations,
* repeat examination procedures,
* deferred examination procedures,
* examination registration,
* medical certificate requirements,
* result appeals,
* examination misconduct.

### Modules

Examples include:

* module descriptions,
* module credits,
* prerequisites,
* module registration,
* assessment weightings,
* module progression,
* repeating modules.

### Student Services

Examples include:

* library services,
* IT support,
* career guidance,
* internship support,
* counselling information,
* student letters,
* student support offices.

### Academic Regulations

Examples include:

* attendance requirements,
* late submissions,
* extensions,
* academic appeals,
* plagiarism,
* academic misconduct,
* mitigating circumstances,
* progression rules.

---

## RAG Knowledge-Base Rules

The RAG knowledge base must contain university information documents, not project-management documents.

Do not add the following files to the RAG vector store:

* `AGENTS.md`
* `PROJECT_SCOPE.md`
* `README.md`
* source-code documentation
* developer notes

These files explain how to build the application. They are not sources for answering student questions.

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

Preserve useful document metadata whenever possible, including:

* document title,
* category,
* filename,
* page number,
* section heading,
* document version or date.

Never place confidential student information in the knowledge base.

---

## Answering and Retrieval Requirements

When the RAG pipeline is implemented, it must:

* answer using retrieved document evidence;
* display the supporting document source;
* avoid inventing policies or procedures;
* clearly state when sufficient information cannot be found;
* distinguish informational guidance from official university decisions;
* recommend contacting the appropriate university department when formal confirmation is required.

The chatbot must never claim that it can:

* approve an appeal;
* approve an extension;
* change examination registrations;
* register a module;
* access private marks;
* make an official academic decision.

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
* avoid placing the entire application in one large file.

Do not leave unexplained placeholder code in completed milestones.

Do not silently ignore errors.

When an error occurs, provide a useful message explaining:

* what failed,
* the likely cause,
* what the user can do next.

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
│   └── __init__.py
│
├── data/
│   └── .gitkeep
│
└── tests/
    └── __init__.py
```

Do not create every possible RAG module before it is needed.

Add modules such as `document_loader.py`, `text_splitter.py`, `embeddings.py`, `retriever.py`, and `rag_pipeline.py` only during their relevant milestones.

---

## Secrets and Security

Never hard-code:

* API keys,
* access tokens,
* passwords,
* private database credentials.

Use environment variables.

Provide a `.env.example` file containing placeholder variable names only.

The real `.env` file must be excluded through `.gitignore`.

Do not print secrets to the terminal or Streamlit interface.

Do not commit uploaded documents unless they are public, fictional, or explicitly approved for the project.

---

## Dependency Rules

Before adding a dependency:

1. Explain what the dependency does.
2. Explain why the current milestone requires it.
3. Check whether an existing dependency already provides the same function.
4. Add it to `requirements.txt`.
5. Avoid unnecessary packages.

For the first Streamlit milestone, use only the minimum dependencies required.

Do not install multiple libraries that perform the same role without a documented reason.

---

## Testing and Validation

After changing code:

1. Check that imports work.
2. Check that the application starts.
3. Run available tests.
4. Report any test or validation failures.
5. Do not claim that something works unless it was validated.

For each completed task, report:

* files created;
* files modified;
* commands run;
* validation results;
* remaining limitations;
* the recommended next milestone.

---

## Change Management

Before making major architectural changes:

1. explain the proposed change;
2. explain why it is needed;
3. identify any new dependencies;
4. identify affected files;
5. keep the change limited to the current milestone.

Do not refactor unrelated parts of the project during a focused task.

Do not delete user-created files without explicit permission.

Do not overwrite `PROJECT_SCOPE.md` unless specifically requested.

---

## Communication Style

Use clear, beginner-friendly explanations.

When completing a task:

* summarize what was implemented;
* explain important code sections;
* explain how to run and test the result;
* mention what has not been implemented yet;
* recommend only one logical next step.

Do not describe the project as complete while planned RAG features are still missing.
