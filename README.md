# Student Support Copilot

**A RAG-Based University Assistance Chatbot**

Student Support Copilot is a university student-support project that will
eventually answer questions using approved university documents. The supported
knowledge areas are Examinations, Modules, Student Services, and Academic
Regulations.

The project is being built one RAG step at a time. It is not yet a complete RAG
chatbot: Milestone 3 loads, splits, and inspects one text-based PDF, but it does
not search documents or generate answers from them.

## Milestone status

- **Milestone 1 — Completed:** Project structure and basic Streamlit chat interface
- **Milestone 2 — Completed:** Page-based PDF document loading with LangChain
- **Milestone 3 — Completed:** Text splitting with LangChain
- **Milestone 4 — Not started:** Embeddings

## What LangChain does in Milestone 2

LangChain is a framework that provides standard building blocks for RAG
applications. In this milestone, only its document-loading interface is used.
No LangChain retrieval, model, agent, or chain has been added.

`PyPDFLoader` reads a PDF from a local path and returns one LangChain `Document`
object for each page. A `Document` keeps the extracted text and its source details
together:

```python
Document(
    page_content="Text extracted from one PDF page",
    metadata={
        "source": "student_guide.pdf",
        "filename": "student_guide.pdf",
        "page": 1,
        "category": "student_services",
    },
)
```

- `page_content` contains machine-readable text extracted from the page.
- `metadata` identifies the source, filename, human-readable page number, and
  selected knowledge category.

The loader uses `pypdf` locally. Uploaded documents are not sent to an external
AI provider and are not permanently saved by the inspection interface.

The document-loading stage is:

```text
PDF -> PyPDFLoader -> page-level LangChain Documents
```

## Text chunking in Milestone 3

Text chunking divides page text into smaller, more focused LangChain `Document`
objects. Page-level loading preserves source boundaries, but a page can contain
several topics and may be too broad for accurate retrieval. Chunking prepares the
text for the later embedding stage:

```text
Page Documents -> RecursiveCharacterTextSplitter -> retrieval-friendly chunks
```

`RecursiveCharacterTextSplitter` tries to split text at readable boundaries such
as paragraphs, lines, and spaces before falling back to individual characters.
The initial experiment uses:

- Chunk size: 800 characters
- Chunk overlap: 120 characters

Chunk size is the target maximum length. Chunk overlap repeats some boundary text
between neighboring chunks so related context is less likely to be separated.
These values are starting settings, not proven optimal settings, and should be
evaluated during later retrieval testing.

Every chunk preserves its page Document metadata, including `source`, `filename`,
`page`, and `category`. It also receives a deterministic `chunk_id`, such as:

```text
examinations_mock_examinations_policy_page_1_chunk_0
```

Chunking is necessary before embeddings because later semantic search should
compare a question with focused sections rather than entire, multi-topic pages.

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- Git (recommended for version control)

## Windows setup

Open PowerShell in the project folder.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the automated tests

```powershell
python -m unittest discover -s tests -v
```

### 5. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` in the default browser. Press
`Ctrl+C` in PowerShell to stop it.

## Using the PDF inspection tool

1. Open **Development tool: inspect one PDF** in the Streamlit application.
2. Select one of the four document categories.
3. Upload one text-based `.pdf` file.
4. Select **Inspect PDF**.
5. Review the filename, category, page counts, short preview, and first-page
   metadata.
6. Optionally enable **Also split and inspect sample chunks** before selecting
   **Inspect PDF** to view chunk counts, settings, IDs, metadata, and short sample
   previews. At most three sample chunks are displayed.

The PDF is copied to a secure temporary path because `PyPDFLoader` needs a local
file path. The temporary copy is deleted after processing, including when loading
fails. The complete extracted document is not displayed by default.

## Knowledge-document locations

Approved source documents should eventually be placed in their matching folders:

```text
documents/raw/
|-- examinations/
|-- modules/
|-- student_services/
`-- academic_regulations/
```

Only use public, fictional, test-specific, or explicitly approved documents.
Never add confidential student records. Project files such as `README.md`,
`AGENTS.md`, and `PROJECT_SCOPE.md` are not knowledge-base sources.

The repository currently includes four fictional development PDFs:

- `documents/raw/examinations/mock_examinations_policy.pdf`
- `documents/raw/modules/mock_modules_handbook.pdf`
- `documents/raw/student_services/mock_student_services_guide.pdf`
- `documents/raw/academic_regulations/mock_academic_regulations.pdf`

These files are safe test data created for this project. They are not official
university policies and must not be presented as real institutional guidance.

## Project structure

```text
student-support-copilot/
|-- app.py
|-- requirements.txt
|-- .env.example
|-- documents/
|   |-- raw/
|   |   |-- examinations/
|   |   |-- modules/
|   |   |-- student_services/
|   |   `-- academic_regulations/
|   `-- processed/
|-- src/
|   |-- __init__.py
|   |-- document_loader.py
|   `-- text_splitter.py
|-- data/
`-- tests/
    |-- __init__.py
    |-- test_document_loader.py
    `-- test_text_splitter.py
```

## Current limitations

- Only text-based PDFs are supported.
- Scanned or image-only PDFs have no machine-readable text and are rejected.
- OCR is not implemented.
- Password-protected, damaged, unsupported, and empty PDFs cannot be loaded.
- Chunk size and overlap are experimental and have not been retrieval-evaluated.
- There are no embeddings, vector store, retrieval, or relevance search.
- No LLM is connected, so the chatbot still returns its safe placeholder response.
- There are no generated answers or source citations yet.
- Authentication, agents, Docker, and deployment are not implemented.
