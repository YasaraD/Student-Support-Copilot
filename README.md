# Student Support Copilot

**A RAG-Based University Assistance Chatbot**

Student Support Copilot is a university student-support project that will
eventually answer questions using approved university documents. The planned
knowledge areas are Examinations, Modules, Student Services, and Academic
Regulations.

Milestone 1 provides only the project structure and a basic Streamlit chat
interface. The assistant currently returns a temporary message because no RAG
knowledge base or AI model has been connected.

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- Git (recommended for version control)

## Windows setup

Open PowerShell in the project folder and follow these steps.

### 1. Create a virtual environment

A virtual environment keeps this project's Python packages separate from other
Python projects on your computer.

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current terminal
session and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit should open the application in your default browser. If it does not,
open the local URL shown in the terminal, usually `http://localhost:8501`.

Press `Ctrl+C` in the terminal to stop the application.

## Milestone 1 status

Implemented:

- Initial project folder structure
- Streamlit title, subtitle, and introductory message
- Display of the four supported categories
- Chat input and chat message display
- Conversation history during the current browser session
- A button to clear the current conversation
- A temporary response explaining that the knowledge base is not connected

Not implemented yet:

- Document loading or text extraction
- Text chunking or embeddings
- Vector storage or retrieval
- AI/LLM integration or RAG answer generation
- Source citations
- Authentication, agents, Docker, or deployment

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
|   `-- __init__.py
|-- data/
`-- tests/
    `-- __init__.py
```

Do not place confidential student information in `documents/raw/`. Only public,
fictional, or explicitly approved university documents should be used in later
milestones.
