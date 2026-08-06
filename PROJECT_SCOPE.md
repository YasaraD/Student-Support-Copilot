# Project Scope: University Student Support RAG Chatbot

## 1. Proposed Project Title

**A RAG-Based University Student Support Chatbot Using Streamlit**

A more advanced title for later use:

**A Multilingual Retrieval-Augmented Generation Chatbot for University Student Support**

---

## 2. Project Overview

The project will develop a chatbot that helps university students find accurate information related to:

1. Examinations
2. Modules
3. Student Services
4. Academic Regulations

The chatbot will use a Retrieval-Augmented Generation, or RAG, system.

Instead of answering only from the general knowledge of a Generative AI model, the system will search relevant university documents and use the retrieved information to generate an answer.

The chatbot interface will be developed using Streamlit.

---

## 3. Main Problem

University students often need information about examinations, modules, student services, and academic regulations.

However, this information may be spread across:

* student handbooks,
* examination regulations,
* module documents,
* university websites,
* student service guides,
* PDF notices,
* frequently asked question documents.

Students may need to search through several documents or contact university staff to find simple answers.

This can result in:

* difficulty finding the correct information,
* students receiving inconsistent answers,
* repeated questions being sent to university staff,
* delays in receiving support,
* confusion caused by long or complicated policy documents.

The proposed chatbot will allow students to ask questions in natural language and receive answers based on official university documents.

---

## 4. Main Project Objective

To design and develop a Streamlit-based RAG chatbot that provides accurate, relevant, and source-supported answers to student questions related to examinations, modules, student services, and academic regulations.

---

## 5. Specific Objectives

The project aims to:

1. Collect and organize university documents related to the four selected categories.

2. extract text from PDF, text, or document files.

3. divide large documents into smaller text sections called chunks.

4. convert document chunks into numerical representations called embeddings.

5. store the embeddings in a vector database.

6. retrieve the most relevant document sections for each student question.

7. use a Generative AI model to generate an answer based on the retrieved information.

8. display the answer through a Streamlit chatbot interface.

9. show the document source used to generate the answer.

10. reduce unsupported or invented answers by instructing the chatbot to answer only from the provided university documents.

---

# 6. Target Users

The main users of the system are:

* undergraduate students,
* postgraduate students,
* new students,
* repeat students,
* students searching for university procedures and regulations.

An administrator role may be added later to upload, remove, or update documents.

---

# 7. Scope of the Four Categories

## Category 1: Examinations

The chatbot will answer questions related to university examinations.

### Included topics

* examination registration,
* examination dates and schedules,
* examination rules,
* examination eligibility,
* repeat examinations,
* deferred examinations,
* missed examinations,
* medical certificates,
* examination admission requirements,
* examination misconduct,
* result-release procedures,
* result appeals,
* resit and retake procedures,
* special examination arrangements.

### Example questions

* How do I register for a repeat examination?
* What happens when I miss an examination?
* When should I submit a medical certificate?
* Can I sit for an examination without completing the required attendance?
* How can I appeal an examination result?
* What is considered examination misconduct?
* Am I eligible for a deferred examination?
* Where can I find my examination timetable?

### Initial limitation

The chatbot will provide examination-related information but will not:

* change examination registrations,
* provide private examination results,
* predict grades,
* make decisions about examination appeals,
* approve medical requests.

These actions require access to official university systems and authorized staff.

---

## Category 2: Modules

The chatbot will answer questions related to university modules and academic study requirements.

### Included topics

* module descriptions,
* module credits,
* prerequisites,
* core modules,
* optional modules,
* module registration,
* module withdrawal,
* module assessment methods,
* coursework and examination weightings,
* module learning outcomes,
* progression requirements,
* repeating a module,
* module completion requirements,
* module coordinator information, where available in the documents.

### Example questions

* What are the prerequisites for this module?
* How many credits is this module worth?
* Is this a core or optional module?
* What percentage of the module mark comes from coursework?
* Can I withdraw from a module?
* What happens when I fail a module?
* Can I take this module before completing its prerequisite?
* What modules must I complete before progressing to the next level?

### Initial limitation

The chatbot will not:

* register students for modules,
* modify official module selections,
* calculate official degree classifications,
* access private student marks,
* replace advice from an academic coordinator.

---

## Category 3: Student Services

The chatbot will answer questions about services available to university students.

### Included topics

* academic support,
* counselling services,
* career guidance,
* internship support,
* library services,
* IT support,
* student letters and document requests,
* student ID services,
* financial support information,
* student clubs and activities,
* disability and accessibility support,
* complaint procedures,
* appointment and contact information,
* available student-support offices.

### Example questions

* How can I contact the student support office?
* How do I request a student confirmation letter?
* What counselling services are available?
* Who should I contact when I cannot access the student portal?
* How can I get internship support?
* What services are available through the university library?
* How can I report an IT problem?
* Where can I find career guidance?

### Initial limitation

The first version will only provide information about these services.

It will not initially:

* book appointments,
* create support tickets,
* send emails,
* approve financial assistance,
* access confidential student records.

These can later be introduced as agent-based features.

---

## Category 4: Academic Regulations

The chatbot will explain official academic rules and university procedures.

### Included topics

* attendance requirements,
* academic progression,
* assessment regulations,
* assignment submission rules,
* late submission procedures,
* extensions,
* academic misconduct,
* plagiarism,
* appeals,
* mitigating circumstances,
* suspension of studies,
* withdrawal from studies,
* degree completion requirements,
* credit requirements,
* classification rules,
* student responsibilities.

### Example questions

* What is the minimum attendance requirement?
* What happens when I submit an assignment late?
* How do I apply for an assignment extension?
* What is considered plagiarism?
* What happens if I fail several modules?
* What are the requirements for progressing to the next academic year?
* How can I submit an academic appeal?
* Can I temporarily suspend my studies?
* How many credits are required to complete the degree?

### Initial limitation

The chatbot will explain academic regulations but will not make official decisions.

For example, it cannot decide whether:

* an appeal should be accepted,
* a student can progress,
* an extension should be approved,
* misconduct has occurred,
* mitigating circumstances are valid.

It should direct students to the appropriate university authority when a formal decision is required.

---

# 8. Functional Scope of the First Version

The first version of the chatbot will support the following functions.

## 8.1 Document Processing

The system will allow selected university documents to be added to the knowledge base.

The system will:

* read PDF or text-based documents,
* extract their text,
* remove unnecessary formatting where possible,
* divide documents into smaller chunks,
* attach metadata to each chunk.

Metadata may include:

* document name,
* category,
* page number,
* section heading,
* date or version,
* source type.

---

## 8.2 Student Question Input

Students will enter questions through a Streamlit chat interface.

For example:

> What happens if I miss an examination because I am sick?

---

## 8.3 Information Retrieval

The system will search the document collection and find the sections most relevant to the question.

For example, it may retrieve sections from:

* Examination Regulations,
* Medical Certificate Guidelines,
* Student Handbook.

---

## 8.4 Answer Generation

The Generative AI model will receive:

* the student's question,
* the retrieved document sections,
* instructions to answer only from the available evidence.

It will then generate a clear and student-friendly response.

---

## 8.5 Source Display

Each answer should display its source.

Example:

> Students who miss an examination due to illness must submit the required medical documentation within the period specified by the university.

**Source:** Examination Regulations, Section 6.2, Page 14

This will help students verify the answer.

---

## 8.6 Conversation History

The chatbot should maintain the current conversation so that students can ask follow-up questions.

Example:

**Student:** What happens if I miss an examination?

**Chatbot:** You may need to submit a medical or mitigating-circumstances request.

**Student:** How many days do I have to submit it?

The system should understand that the second question is connected to the first one.

---

## 8.7 Unknown-Answer Handling

When the required information cannot be found in the documents, the chatbot should not invent an answer.

It should respond with a message such as:

> I could not find enough information in the available university documents to answer this question confidently. Please contact the relevant university department for confirmation.

---

# 9. Features Outside the First Version

The following features will not be part of the initial chatbot:

* direct access to real student records,
* examination result retrieval,
* module registration,
* automatic appointment booking,
* support-ticket creation,
* email sending,
* automatic policy modification,
* payment processing,
* access to confidential university databases,
* fully autonomous agents,
* voice-based interaction,
* mobile application development.

These may be added later after the basic RAG chatbot works correctly.

---

# 10. First-Version System Flow

The initial system will follow this process:

```text
Student asks a question
          ↓
Streamlit receives the question
          ↓
The question is converted into an embedding
          ↓
The vector database searches university documents
          ↓
Relevant document chunks are retrieved
          ↓
The question and retrieved chunks are sent to the AI model
          ↓
The model generates an answer
          ↓
Streamlit displays the answer and its sources
```

---

# 11. Initial Technology Scope

## Streamlit

Streamlit will be used to create the chatbot interface.

It will handle:

* the chat input,
* chatbot messages,
* conversation history,
* source display,
* document upload interface, if included,
* feedback buttons, if included.

## Python

Python will be the main programming language used to:

* process documents,
* create embeddings,
* perform retrieval,
* connect to the Generative AI model,
* manage the RAG pipeline,
* connect the chatbot to the vector database.

## RAG Framework

A framework such as LangChain or LlamaIndex may later be selected to simplify document loading, chunking, retrieval, and model integration.

Only one framework needs to be used initially.

## Vector Database

A vector database such as Qdrant, Chroma, or FAISS will be used to store document embeddings.

For the first prototype, a simple local vector store may be used before moving to a more advanced database such as Qdrant.

## Generative AI Model

An API-based or local language model will generate answers using the retrieved university information.

The model will not be trained from the beginning.

---

# 12. Proposed Minimum Viable Product

The first working version should contain:

* a Streamlit chatbot interface,
* at least one or two documents for each category,
* document text extraction,
* document chunking,
* embedding generation,
* a vector store,
* retrieval of relevant chunks,
* answer generation,
* document-source citations,
* conversation history,
* a safe response when information cannot be found.

The four categories should be clearly tagged in the knowledge base:

```text
Examinations
Modules
Student Services
Academic Regulations
```

---

# 13. Example Project Dataset

The initial document collection may contain:

## Examinations

* Examination Regulations
* Repeat Examination Guidelines
* Examination Timetable Instructions
* Medical and Deferred Examination Guidelines

## Modules

* Module Descriptors
* Programme Handbook
* Module Registration Guide
* Progression Requirements

## Student Services

* Student Support Guide
* Library Services Guide
* Career and Internship Guide
* IT Support Guide

## Academic Regulations

* Academic Regulations Handbook
* Attendance Policy
* Academic Misconduct Policy
* Appeals and Mitigating Circumstances Policy

Only documents that can legally and ethically be used should be added to the system.

---

# 14. Project Success Criteria

The first version will be considered successful when:

1. students can enter questions through Streamlit;

2. the system retrieves information from the correct category;

3. answers are based on the provided documents;

4. sources are displayed with the answers;

5. the system avoids answering when there is insufficient evidence;

6. follow-up questions work within the current conversation;

7. the chatbot can correctly answer a prepared test set covering all four categories;

8. the system is easy for a student to use without technical knowledge.

---

# 15. Final Scope Statement

This project will develop a Streamlit-based RAG chatbot that supports university students by answering questions related to examinations, modules, student services, and academic regulations.

The system will retrieve relevant information from approved university documents and use a Generative AI model to produce clear, source-supported answers.

The first version will focus only on information retrieval and question answering. It will not access private student records or perform official university actions. Agent-based functions, such as creating support tickets or booking appointments, may be added as later extensions after the core RAG chatbot has been successfully implemented and evaluated.
