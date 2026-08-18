# Software Requirements Document
## Credit Risk Analyzer — Retrieval-Augmented Generation (RAG)

| | |
|---|---|
| **Project** | Credit Risk Analyzer (RAG) |
| **Author** | Srikala Gangi Reddy |
| **Version** | 1.0 (Draft) |
| **Date** | 7 July 2026 |
| **Status** | For build |

---

## 1. Purpose

This document defines the requirements for a **Credit Risk Analyzer application** that lets a user upload a financial document — a 10-K, a bond prospectus, or an earnings transcript — ask natural-language questions about it, and receive answers that are **grounded in the document and supported by page-level citations**.

The application uses a Retrieval-Augmented Generation (RAG) approach: rather than relying on a language model's memory, it retrieves the most relevant passages from the uploaded document and asks the model to answer using only those passages. This keeps answers accurate, current to the specific document, and auditable — properties that matter in a financial-research context.

## 2. Background and rationale

The application is the flagship project of a domain-focused AI portfolio. It deliberately mirrors real fixed-income credit-research work — where analysts must extract specific facts, figures, and risk factors from long, dense filings — reframed with modern LLM engineering. The goal is to demonstrate the combination of financial-domain knowledge and current AI/cloud skills in a single, demonstrable artifact.

## 3. Objectives and success criteria

The project is successful when:

1. A user can upload a text-based financial PDF and ask questions about it through a web interface.
2. Answers are grounded in the document and cite the page(s) used.
3. When a question cannot be answered from the document, the system says so rather than fabricating an answer.
4. The application is **deployed to a public URL** (Hugging Face Spaces) and demoable end-to-end in under two minutes.
5. The codebase is modular, documented, and explainable component-by-component in an interview.

## 4. Scope

### 4.1 In scope (MVP)
- Single-document upload and Q&A per session.
- Text-based PDF documents (10-K, bond prospectus, earnings transcript).
- Natural-language questions and grounded, cited answers.
- A simple web UI with conversation history.
- Local execution and public cloud deployment.

### 4.2 Out of scope (MVP — see §11 roadmap)
- Multiple documents or cross-document comparison in one session.
- Scanned/image-only PDFs requiring OCR.
- User accounts, authentication, or persistent storage of documents.
- Non-PDF formats (Word, HTML, plain text).
- Languages other than English.

## 5. Users and personas

**Primary persona — Research / Credit Analyst.** Reviews long financial filings and needs to locate specific facts (revenue, debt maturities, risk factors, covenants) quickly and trace them back to the source. Comfortable reading financial documents; not necessarily technical.

**Secondary persona — Reviewer / Hiring Manager.** Evaluates the application as a portfolio piece: cares about correctness, grounding, clean design, and whether the author can explain the engineering choices.

## 6. Functional requirements

Each requirement is testable and traceable.

### 6.1 Document ingestion
- **FR-1** The user can upload a single PDF document via the interface.
- **FR-2** The system extracts text from every page, preserving the page number of each passage.
- **FR-3** The system splits extracted text into overlapping chunks, retaining each chunk's source page.
- **FR-4** The system supports multi-page documents (hundreds of pages).
- **FR-5** If a PDF contains no extractable text (e.g. a scanned image), the system returns a clear, friendly message explaining that OCR would be required, rather than failing silently.

### 6.2 Indexing
- **FR-6** The system converts chunks into vector embeddings and builds a searchable index in memory.
- **FR-7** The system reports when indexing is complete, including how many chunks were indexed.

### 6.3 Question answering
- **FR-8** The user can enter a free-text question about the uploaded document.
- **FR-9** The system retrieves the top-k most relevant chunks for each question (k configurable).
- **FR-10** The system generates an answer using **only** the retrieved content.
- **FR-11** Each answer includes the page number(s) of the source passages used (e.g. *[p. 12]*).
- **FR-12** If the answer is not present in the retrieved content, the system explicitly states it cannot find the answer in the document.
- **FR-13** The system displays which page(s)/excerpts the answer was drawn from.
- **FR-14** The user can ask multiple questions against the same indexed document without re-uploading.

### 6.4 Interface
- **FR-15** A web interface provides: document upload, indexing status, a question input, an answer display, and a running conversation history.

### 6.5 Credit metrics extraction
- **FR-16** The user can extract credit metrics (profitability, leverage, coverage, liquidity, cash flow, capital return, debt maturity schedule) from the uploaded filing via a dedicated action, separate from Q&A.
- **FR-17** Metrics extraction uses **direct long-context extraction** — the parsed filing text is sent to the chat model in a single call to read off raw, as-reported line items — **not** the embeddings/RAG path used for Q&A. This avoids the free-tier embedding rate limit that a full-filing index would hit for the ~30 line items metrics need, and lets the model see all relevant figures at once rather than only the top-k retrieved chunks.
- **FR-18** The language model extracts only raw reported figures; it never computes a ratio, sum, or any derived value. All derived metrics are computed in application code.
- **FR-19** Every derived metric carries a citation (source statement + page number) back to the line item(s) it was computed from.
- **FR-20** If a line item required for a derived metric is not available in the filing, that metric — and anything computed from it — is flagged "not available" with a reason. It is never computed by substituting a placeholder value such as zero.

## 7. Non-functional requirements

- **NFR-1 — Grounding / accuracy.** The system must minimise hallucination; every answer must be traceable to a cited source passage. This is the single most important quality attribute for a financial tool.
- **NFR-2 — Performance.** For a typical document, a question should return an answer within a few seconds. Indexing time should scale reasonably with document length.
- **NFR-3 — Usability.** A non-technical financial user should be able to go from upload to answer without instructions.
- **NFR-4 — Security & privacy.** API keys are supplied via environment variables / platform secrets and never committed to source control. Uploaded documents are not persisted beyond the session. Users are advised not to upload confidential material to the public demo.
- **NFR-5 — Portability & deployability.** The application runs locally and deploys to Hugging Face Spaces with minimal dependencies and no specialised infrastructure.
- **NFR-6 — Maintainability & modularity.** Ingestion, embeddings, vector store, and generation are separated so any component (e.g. the embeddings provider or vector store) can be swapped without rewriting the app.
- **NFR-7 — Cost efficiency.** The system defaults to cost-efficient models for both embeddings and generation, keeping per-query cost low.
- **NFR-8 — Reliability.** Errors (bad file, missing key, empty document, API failure) are handled gracefully with user-facing messages, not stack traces.
- **NFR-9 — Configurability.** Chunk size, overlap, top-k, and model names are configurable in one place.

## 8. Data requirements

- **Input:** a single text-based PDF, typically 1–300 pages.
- **Provenance:** each chunk carries its page number and source filename through the whole pipeline to enable citations.
- **Transient data:** extracted text and chunks exist in memory for the session, while embeddings are persisted to disk in a configurable cache for reuse across runs of the same document/configuration.
- **No training data:** the system does not train or fine-tune any model; it uses pre-trained embedding and chat models at inference time.

## 9. Representative use cases

- **UC-1 — Risk factors.** Analyst uploads a 10-K and asks, *"What are the key risk factors?"* → receives a concise summary citing the relevant pages.
- **UC-2 — Specific figure.** Analyst asks, *"What was net revenue for the year?"* → receives the figure with a page citation.
- **UC-3 — Covenant / maturity lookup.** Analyst uploads a bond prospectus and asks about debt maturities or covenants → receives the answer grounded in the prospectus.
- **UC-4 — Not-in-document.** Analyst asks something the document does not cover → system replies that the answer is not in the document.

## 10. Assumptions and constraints

- Uploaded PDFs are text-based (not scanned images) for the MVP.
- One document is analysed per session.
- An internet connection and a valid model-provider API key are available at runtime.
- Content is in English.
- Very large documents may exceed practical indexing/context limits and are handled by retrieval (only the top-k passages reach the model).
- The application is for research and educational use and does not constitute financial advice.

## 11. Future enhancements (v2 roadmap)

- Multi-document support with per-source attribution and cross-document comparison.
- OCR fallback for scanned PDFs.
- Highlight the exact source sentence, not just the page.
- Option to run fully offline using a local embedding model.
- A small evaluation set (question/answer pairs) to measure retrieval and answer quality.
- Support for additional formats (Word, HTML, plain text).

## 12. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Model hallucination on figures | Wrong financial answer | Strict grounding prompt + citations + explicit "not found" behaviour (FR-10 to FR-12) |
| Scanned/image PDFs | No text to index | Detect and message clearly (FR-5); OCR in v2 |
| Retrieval misses relevant passage | Incomplete answer | Tunable chunking and top-k (NFR-9); evaluation set in v2 |
| API cost on heavy use | Running cost | Cost-efficient default models (NFR-7) |
| Very long documents | Context/index limits | Retrieval limits context to top-k passages |

## 13. Acceptance criteria

The MVP is accepted when all of the following hold:

1. A text-based financial PDF can be uploaded and successfully indexed, with a completion status shown. *(FR-1, FR-2, FR-6, FR-7)*
2. A question returns an answer grounded in the document with at least one correct page citation. *(FR-9 to FR-11, FR-13)*
3. A question with no answer in the document produces an explicit "not found" response. *(FR-12)*
4. Multiple questions can be asked against one uploaded document. *(FR-14)*
5. An unsupported or empty PDF produces a clear, friendly error. *(FR-5, NFR-8)*
6. The application runs locally and is deployed to a public URL. *(NFR-5)*
7. No secrets are present in the source code. *(NFR-4)*

## 14. Glossary

- **RAG (Retrieval-Augmented Generation):** answering a question by first retrieving relevant source passages, then having a language model generate an answer from them.
- **Embedding:** a numeric vector representing the meaning of a piece of text, enabling similarity search.
- **Chunk:** a small, retrievable segment of the source document.
- **Vector store:** a structure that holds embeddings and finds the most similar ones to a query.
- **Top-k:** the number of most-relevant chunks retrieved for a question.
- **Grounding:** constraining the model to answer only from provided source content.
- **Hallucination:** a fluent but unsupported or incorrect model output.
- **Citation:** a reference (here, a page number) pointing to the source of an answer.
