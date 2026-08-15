# Student Support Copilot

**A RAG-Based University Assistance Chatbot**

Student Support Copilot is a university student-support project that will
eventually answer questions using approved university documents. The supported
knowledge areas are Examinations, Modules, Student Services, and Academic
Regulations.

The project is being built one RAG step at a time. Milestone 7 provides the first
complete 2-Step RAG path: it retrieves relevant chunks, sends the current
question and those chunks to Gemini, and displays a generated answer with
deterministic sources. It remains a student prototype, not an official service.

## Milestone status

- **Milestone 1 — Completed:** Project structure and basic Streamlit chat interface
- **Milestone 2 — Completed:** Page-based PDF document loading with LangChain
- **Milestone 3 — Completed:** Text splitting with LangChain
- **Milestone 4 — Completed:** Local dense embeddings with BGE-M3
- **Milestone 5 — Completed:** Persistent local Chroma vector store
- **Milestone 6 — Completed:** Dense semantic retrieval and validation
- **Milestone 7 — Completed:** Gemini integration and first 2-Step RAG pipeline
- **Operational index management — Completed:** Central configuration, manifests,
  document-change detection, safe staging rebuilds, and one-generation rollback

## Embeddings in Milestone 4

An embedding is a numerical representation of a piece of text. Texts with
similar meanings should produce vectors that point in similar directions. This
allows the retrieval system to compare a student question with document
chunks mathematically instead of relying only on matching exact words.

An embedding model is different from a Large Language Model (LLM):

- The embedding model converts text into vectors for semantic search.
- An LLM generates and explains natural-language answers.

Gemini is now connected separately for answer generation in Milestone 7.

### Selected embedding model

The initial embedding model for the RAG system is:

`BAAI/bge-m3`

The project accesses it through LangChain's Hugging Face integration:

`HuggingFaceEmbeddings` from `langchain_huggingface`.

The initial implementation uses only BGE-M3 dense embeddings.

It does not use BGE-M3 sparse retrieval, ColBERT-style multi-vector retrieval,
hybrid retrieval, or reranking.

The initial embedding workflow is:

```text
LangChain Document chunk
        ↓
page_content
        ↓
BAAI/bge-m3
        ↓
1024-dimensional dense vector
```

The complete implemented processing pipeline is:

```text
PDF
-> PyPDFLoader
-> page Documents
-> RecursiveCharacterTextSplitter
-> chunks
-> BAAI/bge-m3
-> normalized 1024-dimensional dense embeddings
-> persistent Chroma collection
-> Retriever
-> relevant chunks
```

BGE-M3 was selected because it provides multilingual text embeddings and can run
locally. Only its normal dense-vector output is used. Sparse embeddings, hybrid
retrieval, ColBERT-style multi-vector retrieval, and reranking remain outside
this milestone.

Student queries are embedded using the same model as the document
chunks, because vectors from different model spaces cannot be compared reliably.

The implementation requests normalized embeddings, which scale each vector to
unit length while preserving its semantic direction. The model name is defined
once in the `EMBEDDING_MODEL_NAME` configuration constant.

`HuggingFaceEmbeddings` provides LangChain's standard query/document embedding
interface. The underlying `sentence-transformers` package handles local model
loading, tokenization, batching, and numerical vector generation.

The embedding module must remain separate from:

* document loading;
* text splitting;
* vector storage;
* retrieval;
* LLM generation.

The first execution may download several gigabytes of model files from Hugging
Face and can take several minutes depending on the internet connection and
computer. Later executions reuse Hugging Face's disk cache. The application also
keeps one loaded model object in process memory, avoiding repeated model loading
during ordinary Streamlit reruns. This resource cache does not store student
questions, uploaded document text, or generated vectors.

No embedding API key is required for the local BGE-M3 implementation.

Embeddings are stored persistently in the local Chroma development index and can
now be searched through the Milestone 6 retriever.

## Vector store in Milestone 5

A vector store is a database designed to keep embedding vectors together with
their original text, identifiers, and metadata. BGE-M3 creates vectors; Chroma
stores and indexes them. The original approved PDF documents remain the source
of university truth—Chroma contains generated representations that can be
rebuilt from those sources.

Chroma was selected for the first prototype because it offers simple local
persistence, metadata storage, deterministic IDs, and a dedicated LangChain
integration. FAISS would require more surrounding metadata and persistence code.
Qdrant is better suited to larger hosted or multi-user systems, which this local
prototype does not currently require.

A Chroma collection is a named group of related text, vectors, IDs, and metadata.
This project uses:

```text
Collection: student_support_knowledge
Directory:  data/chroma_db/
Distance:   cosine
```

Cosine distance compares vector directions. Semantically similar normalized
embeddings should have a smaller cosine distance. Milestone 6 uses this
configuration for dense question-to-chunk search.

Every chunk's deterministic `chunk_id` is also used as its Chroma document ID.
The stored metadata includes `source`, `filename`, `page`, `category`, and
`chunk_id`, allowing later results to be traced back to the correct source.

The implemented pipeline is:

```text
PDF
-> PyPDFLoader
-> page Documents
-> RecursiveCharacterTextSplitter
-> chunks
-> BAAI/bge-m3
-> normalized 1024-dimensional embeddings
-> persistent local Chroma collection
-> Retriever
-> relevant chunks
```

`data/chroma_db/` is generated data and is excluded from Git. Rebuild the index
explicitly when source PDFs change, chunk size/overlap changes, the embedding
model changes, or the collection configuration changes. Opening the existing
index does not re-embed or re-index documents.

## Knowledge-index configuration and safe updates

The knowledge index has two different kinds of configuration:

- **Index-defining settings** determine stored chunks and vectors. These include
  source categories, chunk size and overlap, embedding configuration, metadata
  schema, collection name, and distance metric. Changing them requires a rebuild.
- **Retrieval-time settings** control how an existing index is searched. For
  example, changing `TOP_K` does not require re-embedding documents.

`src/config.py` is the single source of these settings. Existing modules import
their constants from `INDEX_CONFIG`, so document loading, splitting, embeddings,
Chroma, and retrieval cannot silently use different defaults.

### Index manifest

After a successful rebuild, the project writes:

```text
data/chroma_db/index_manifest.json
```

The manifest is a machine-readable record of how the active index was produced.
It contains:

- a unique build identifier and UTC build time;
- the complete index-defining configuration and its SHA-256 fingerprint;
- each source PDF's category, relative path, size, and SHA-256 content hash;
- document, page, and chunk counts;
- chunk counts for all four categories.

SHA-256 is used as a content fingerprint. Merely changing a file's timestamp does
not trigger a rebuild, but adding, modifying, moving, or removing a PDF does.

### Safe staging and rollback

The rebuild command never appends directly to the active collection. It follows
this process:

```text
Discover and hash source PDFs
-> compare sources/settings with the active manifest
-> load and split every approved PDF
-> embed chunks into a uniquely named staging collection
-> validate IDs, total count, and category coverage
-> rename the active collection as the rollback generation
-> promote staging as the active collection
-> promote the new manifest
```

If loading, splitting, embedding, storage, or staging validation fails, the
active collection is preserved. After a successful replacement, exactly one
previous collection is retained for rollback. A later successful rebuild replaces
that older rollback generation.

This is a controlled full-rebuild workflow, not incremental indexing. Full
rebuilds are simpler and safer for the project's current small document set.

## Retrieval in Milestone 6

Retrieval is the evidence-finding stage of RAG. A student question is converted
to a query embedding and compared with the stored chunk embeddings. The output
is a ranked list of relevant LangChain `Document` objects; it is not a generated
answer.

A vector store and a retriever have different jobs:

- Chroma is the vector store. It persists vectors, text, IDs, and metadata.
- The retriever is the read-only search interface that accepts a question and
  returns relevant stored Documents.

The implemented query flow is:

```text
Student question
-> BAAI/bge-m3 query embedding
-> existing Chroma collection
-> cosine-distance comparison
-> top relevant chunks
```

The query uses the same `BAAI/bge-m3` configuration that created the indexed
document embeddings. Embeddings from different models occupy different numeric
spaces and cannot be compared reliably.

Semantic similarity searches for related meaning rather than requiring exact
word matches. Top-k retrieval means returning the best `k` candidates. The
initial experimental setting is `TOP_K = 4`, so the chatbot retrieves up to four
chunks for each question. Alternative values remain available in the retriever
module for controlled development experiments.

Optional category filtering limits Chroma to records whose `category` metadata
matches `examinations`, `modules`, `student_services`, or
`academic_regulations`. `None` searches all categories. The system does not infer
a category or run an intent-classification model.

### Retrieval distances

The inspection path uses the current `langchain-chroma`
`similarity_search_with_score` method. Its returned float is a **raw cosine
distance**: lower is a better match. It is not a similarity percentage and is
not converted to one. Results retain their rank, original chunk text, and stored
metadata.

The project also exposes LangChain's standard Retriever interface through
`as_retriever(search_type="similarity")`. Direct scored search is useful for
debugging and evaluation. The standard interface returns Documents without raw
scores and will be useful when composing the future RAG workflow.

Retrieval is deliberately validated before adding an LLM so incorrect evidence
cannot be hidden behind fluent generated text. Vector search always returns the
nearest available chunks when possible. Therefore, unsupported questions can
still receive unrelated nearest neighbours; a retrieved result alone does not
prove that the question is answerable.

### Controlled development evaluation

The controlled dataset contains 10 supported questions and 2 unsupported
questions. It includes all four categories, paraphrases, differently worded
queries, and human-defined expected categories and sources. Against the real
43-chunk BGE-M3/Chroma index, the development results were:

- Top-1 category accuracy: 10/10 (100%)
- Top-4 category hit rate: 10/10 (100%)
- Expected-source hit rate within top 4: 10/10 (100%)
- Category-filter check: all four returned records were from `examinations`
- Stored records before and after evaluation: 43

These are small development metrics for fictional documents, not final research
evaluation results. Both unsupported questions still returned nearest chunks,
including Student Services, Modules, Academic Regulations, and Examinations
records. No code currently decides that those questions lack sufficient
evidence.

## Milestone 7 — LLM Integration and 2-Step RAG

A Large Language Model (LLM) generates readable language from instructions and
input text. This project uses the Google Gemini Developer API with
`gemini-3.5-flash-lite` through LangChain's `ChatGoogleGenerativeAI` interface.
The model name can be changed through `LLM_MODEL` without changing source code.

The components have distinct jobs:

- **BGE-M3** converts document chunks and questions into vectors. It does not
  write answers.
- **Chroma** stores and searches those vectors, text, and metadata. It does not
  generate language.
- **The retriever** finds the top four chunks most related to the current
  question.
- **Gemini** receives the current question and those retrieved chunks, then
  produces a student-friendly answer from that evidence.

This is called **2-Step RAG** because its two major runtime phases are:

1. Retrieval — find relevant evidence.
2. Generation — answer from that evidence.

It was selected instead of agentic RAG because this fixed sequence is easier to
inspect, test, and explain. No autonomous agent chooses tools or changes the
workflow.

The current runtime pipeline is:

```text
Student question
        ↓
BAAI/bge-m3 query embedding
        ↓
Chroma
        ↓
Retriever
        ↓
Top 4 relevant chunks
        ↓
Grounded RAG prompt
        ↓
Gemini 3.5 Flash-Lite
        ↓
Generated answer
        +
Deterministic source metadata
        ↓
Streamlit
```

### Grounding and the RAG prompt

Grounding means requiring the generated answer to stay supported by retrieved
document evidence. The central prompt tells Gemini to use only the supplied
context, ignore instructions found inside document text, avoid inventing policy,
identify conflicting evidence, and avoid claiming official authority.

When the chunks do not contain enough evidence, the prompt requires this reply:

> I could not find enough information in the available university documents to
> answer this question confidently. Please contact the relevant university
> department for official confirmation.

This answerability decision is model-based and is not mathematically guaranteed.
Nearest-neighbour retrieval can return unrelated chunks for unsupported
questions. No fixed distance threshold is used because the current evaluation
set is too small to calibrate one responsibly.

Student-facing sources do not come from Gemini's prose. The application builds
them deterministically from retrieved LangChain `Document.metadata`, including
filename and page, and removes duplicate filename/page pairs while preserving
retrieval order.

### Local and external processing

The following operations remain local:

- PDF processing and chunking;
- BGE-M3 embeddings;
- the persistent Chroma index;
- semantic retrieval.

For each submitted question, the external Gemini API receives:

- the current student question;
- only the selected top retrieved text chunks and limited source metadata.

The complete original PDFs, embeddings, Chroma configuration, previous chat
turns, and API key are not sent in the RAG prompt. Do not enter confidential
student data or use private documents without an approved privacy process.

The chat interface displays prior messages, but every question is retrieved and
answered independently. A follow-up such as “What if I fail it twice?” may not
work because conversation-aware RAG is intentionally not implemented yet.

### Milestone 7 validation status

The prompt, Gemini wrapper, source formatter, RAG orchestration, and Streamlit
answer/source rendering are covered by automated tests with a mocked model. A
controlled run also exercised real BGE-M3 retrieval from the 43-record Chroma
index while replacing only the external generator with a deterministic test
function.

A real Gemini request was not run during the initial Milestone 7 validation
because no local `GOOGLE_API_KEY` was configured. The generation evaluation file
therefore records expected facts and a clear “not run” status rather than
inventing model results. After configuring a key, the five listed questions can
be evaluated manually using only the fictional documents.


## What LangChain does in Milestone 2

LangChain is a framework that provides standard building blocks for RAG
applications. During Milestone 2, only its document-loading interface was used;
retrieval was added separately in Milestone 6. No LLM, agent, or chain has been
added.

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

The loader uses `pypdf` locally. Document-loading behavior is validated through
the automated tests and remains separate from the student-facing interface.

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
These values are starting settings. They worked in the small Milestone 6
development evaluation but are not proven optimal settings.

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

### 4. Configure Gemini locally

Create a local `.env` from the safe example:

```powershell
Copy-Item .env.example .env
```

Open `.env` in a text editor and set:

```text
GOOGLE_API_KEY=your_local_key_here
LLM_MODEL=gemini-3.5-flash-lite
```

Get a Gemini Developer API key from Google AI Studio. Never commit or share the
real value. `.env` is ignored by Git, while `.env.example` contains placeholders
only. If `GOOGLE_API_KEY` is missing, the app displays a configuration message
instead of crashing. Free API access, quotas, rate limits, and model availability
are controlled by Google and may change.

### 5. Run the automated tests

```powershell
python -m unittest discover -s tests -v
```

### 6. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` in the default browser. Press
`Ctrl+C` in PowerShell to stop it.

## Managing the knowledge index

Stop Streamlit before running rebuild or rollback commands. The local Chroma
database should have only one administrative writer.

### Check whether a rebuild is needed

```powershell
python scripts\rebuild_index.py status
```

This read-only command reports the active and rollback collection counts,
manifest build ID, added/modified/removed documents, configuration changes, and
whether a rebuild is required. It does not load BGE-M3.

### Rebuild after changing documents or index settings

```powershell
python scripts\rebuild_index.py rebuild
```

Review the detected changes, then type `yes` when prompted. The first use of
BGE-M3 may download the model; later runs reuse its Hugging Face disk cache but
still load the model into memory for embedding.

For a non-interactive terminal after reviewing `status`:

```powershell
python scripts\rebuild_index.py rebuild --yes
```

Use `--force` only when you intentionally want to rebuild even though document
hashes and index-defining configuration are unchanged:

```powershell
python scripts\rebuild_index.py rebuild --force --yes
```

### Restore the previous index generation

```powershell
python scripts\rebuild_index.py rollback
```

After confirmation, active and backup collection names are swapped. The index
being replaced becomes the new rollback generation, so the operation can be
reversed by running rollback again. If the original collection predates manifest
support, rolling back to it correctly reports that its manifest is unavailable.

After rebuild or rollback, run `status`, restart Streamlit, and test known
questions from all four categories.

## Using the chatbot

1. Review the four supported knowledge areas on the main page.
2. Enter one clear, complete university-support question in the chat box.
3. Wait while the application searches the local knowledge base and asks Gemini
   to prepare a grounded answer. The first question can take longer while the
   local BGE-M3 model loads into memory.
4. Read the answer and open **View supporting sources** to check the source
   filename, category, and page number.
5. Use **Clear chat** to reset the conversation shown in the current browser
   session.

The student-facing page intentionally excludes PDF inspection, chunk previews,
embedding vectors, Chroma rebuild controls, retrieval distances, and raw chunk
identifiers. Those operations remain implemented in separate `src/` modules and
are checked through the automated test suite. The Streamlit application never
rebuilds the index automatically.

Each submitted question is processed independently. Previous messages remain
visible for convenience, but they are not yet sent back to the retriever or LLM
as conversation context.

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
|-- scripts/
|   |-- __init__.py
|   `-- rebuild_index.py
|-- documents/
|   |-- raw/
|   |   |-- examinations/
|   |   |-- modules/
|   |   |-- student_services/
|   |   `-- academic_regulations/
|   `-- processed/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- document_loader.py
|   |-- text_splitter.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- index_manifest.py
|   |-- index_manager.py
|   |-- retriever.py
|   |-- prompts.py
|   |-- llm.py
|   |-- source_formatter.py
|   `-- rag_pipeline.py
|-- evaluation/
|   |-- retrieval_questions.json
|   `-- generation_questions.json
|-- data/
|   `-- chroma_db/          # generated and ignored by Git
`-- tests/
    |-- __init__.py
    |-- test_document_loader.py
    |-- test_config.py
    |-- test_text_splitter.py
    |-- test_embeddings.py
    |-- test_vector_store.py
    |-- test_index_manifest.py
    |-- test_index_manager.py
    |-- test_retriever.py
    |-- test_prompts.py
    |-- test_llm.py
    |-- test_source_formatter.py
    `-- test_rag_pipeline.py
```

## Current limitations

- Only text-based PDFs are supported.
- Scanned or image-only PDFs have no machine-readable text and are rejected.
- OCR is not implemented.
- Password-protected, damaged, unsupported, and empty PDFs cannot be loaded.
- Chunk size, overlap, and `TOP_K` remain experimental despite the small initial
  retrieval evaluation.
- Knowledge-index updates use controlled full rebuilds; incremental per-document
  updates are not implemented.
- Exactly one previous collection is retained for rollback; this is not a full
  backup history.
- Streamlit must be stopped during rebuild and rollback because local embedded
  Chroma is not an administrative multi-writer service.
- Dense semantic retrieval returns nearest chunks but does not determine whether
  they contain enough evidence to answer a question.
- Retrieval distances are raw diagnostic values, not confidence percentages.
- Each question is independent; conversation-aware retrieval and generation are
  not implemented.
- Unsupported-question handling depends on Gemini following the grounded prompt;
  there is no calibrated retrieval-confidence threshold.
- There is no reranking, hybrid retrieval, sparse retrieval, MMR, or query
  rewriting.
- Generation depends on an external Gemini API connection, valid credentials,
  model availability, and provider-controlled quotas or free-tier limits.
- Token streaming is not implemented.
- Authentication, agents, Docker, and deployment are not implemented.
