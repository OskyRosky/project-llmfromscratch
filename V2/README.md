# **LLM From Scratch (V2) — Token Chat (BPE)**

This repository presents a complete end-to-end implementation of a token-level language model, trained from scratch using Byte Pair Encoding (BPE) and later instruction-tuned, exposed through an interactive Streamlit application and fully deployable via Docker.

The goal of this project is not to compete with large-scale foundation models, but to prove, understand, and document the full lifecycle of a language model at the token level: from tokenization and generation to inference control, safety mechanisms, and deployment.

This project demonstrates that a small, well-controlled model can behave honestly, traceably, and predictably, even when its knowledge is intentionally limited.

⸻

Project Objectives

The main objectives of this project are:
	•	To validate that a token-based LLM pipeline works correctly end to end.
	•	To demonstrate instruction-based inference on top of a minimal GPT-style architecture.
	•	To enforce explicit safety and correctness rules instead of relying on implicit model behavior.
	•	To expose inference metadata so every answer can be explained and audited.
	•	To provide a clean, reproducible deployment using Streamlit and Docker.

⸻

What This Model Is (and Is Not)

This model is:
	•	A real token-level language model.
	•	Trained using a custom BPE tokenizer.
	•	Capable of generating responses autoregressively.
	•	Controlled by explicit inference logic and guards.
	•	Designed for learning, experimentation, and demonstration.

This model is not:
	•	A large-scale general knowledge model.
	•	A retrieval-augmented system.
	•	A production-ready assistant.
	•	Designed to “guess” or hallucinate answers.

When the model does not have enough knowledge, it explicitly says so.

⸻

Core Architecture Overview

Tokenizer (BPE)

The model uses a custom Byte Pair Encoding tokenizer, defined by:
	•	A tokenizer configuration.
	•	A vocabulary learned from training data.
	•	A meta file defining special tokens and decoding rules.

All inference happens strictly at the token level, with no word-level shortcuts.

⸻

Language Model

The core model is a GPT-style autoregressive transformer, trained from scratch with:
	•	A small number of layers and heads.
	•	A limited context window.
	•	A deliberately constrained parameter count.

This design keeps inference fast and behavior interpretable.

⸻

Instruction Tuning

After base training, the model is instruction-tuned so it understands prompts of the form:
	•	Instruction
	•	Expected response

This allows the same base model to behave differently depending on how it is prompted, without changing weights at inference time.

⸻

Inference Logic and Guards

One of the most important aspects of this project is that generation is not left alone. Every inference follows a strict decision flow implemented in answer_with_meta().

1. Private Information Guard

If a question asks for personal or private information (e.g. personal preferences, relationships, identities), the model does not attempt generation.

Instead, it returns a fixed refusal stating that it does not have access to personal data.

This is intentional and deterministic.

⸻

2. Verified Fact Anchoring

If the question matches a verified fact (via a lightweight FAQ → FACT mapping), the model:
	•	Is explicitly instructed to answer using that fact.
	•	Is validated against the expected content.
	•	Falls back to the exact fact if the generated output deviates.

This prevents hallucination while still exercising token generation.

⸻

3. Normal Generation (No Fact)

If no verified fact exists and the question is not private, the model generates a response normally using its learned token distributions.

Decoding parameters (top-k, temperature, repetition penalties) are configurable to explore different behaviors.

⸻

4. Unknown / Derail Guard

Some general topics (e.g. advanced science concepts) are known to cause semantic derailment in very small models.

If the model produces an answer that:
	•	Appears confident,
	•	But contains unrelated anchor concepts,

The system blocks the output and returns an honest refusal indicating insufficient knowledge.

⸻

Unified Refusal Strategy

All refusals are normalized into clear, human-readable messages, with explicit metadata indicating why the refusal happened.

This ensures the UI and any downstream system can reason about the response.

⸻

Inference Metadata

Every call to the model returns not only the answer, but also metadata such as:
	•	Whether a verified fact was used.
	•	Whether a private guard was triggered.
	•	Whether an unknown/derail guard was triggered.
	•	The reason for refusal (if any).
	•	The total inference latency in milliseconds.

This makes the system auditable and explainable, even at demo scale.

⸻

Streamlit Application

The Streamlit app provides an interactive interface to the model and exposes:
	•	The generated answer.
	•	A clear status badge explaining how the answer was produced.
	•	Latency information.
	•	Optional display of the fact anchor used during generation.
	•	Full inference metadata for debugging and inspection.

The UI is intentionally simple and focused on transparency.

⸻

Latency Characteristics

Because the model is:
	•	Small,
	•	Running locally,
	•	CPU-based by default,
	•	Generating very short sequences,

Latency is extremely low (often tens of milliseconds). This is expected and desirable for this scale of model.

Fast responses do not mean the model is rule-based or hard-coded; they simply reflect the small architecture and local execution.

⸻

Testing and Stability

A fixed smoke test suite is included to ensure:
	•	Verified facts still work.
	•	Private questions are always blocked.
	•	Unknown topics are handled consistently.
	•	Future changes do not break core guarantees.

This suite is intended to be run before every commit.

⸻

Dockerized Deployment

The entire application can be deployed via Docker, with:
	•	All model assets bundled.
	•	Streamlit exposed on a dedicated port.
	•	Environment variables controlling model paths and device selection.

The container can be run independently of other versions, allowing multiple model variants to coexist without conflict.

⸻

Why This Project Matters

This repository demonstrates something subtle but important:

A language model does not need to know everything to behave correctly.

By combining:
	•	Small-scale modeling,
	•	Explicit inference logic,
	•	Clear refusal policies,
	•	And transparent metadata,

We get a system that is predictable, explainable, and honest.

That foundation is what larger, more capable systems are built upon.

⸻

Future Directions (Optional)

Possible next steps include:
	•	Expanding instruction-tuning datasets.
	•	Adding document-grounded retrieval (RAG).
	•	Training larger token vocabularies.
	•	Introducing evaluation benchmarks.

None of these are required to validate the core goal of this project.

⸻

Status

This project is considered complete for its intended scope:
	•	Token-level LLM ✔
	•	Instruction tuning ✔
	•	Safety guards ✔
	•	Streamlit UI ✔
	•	Docker deployment ✔
	•	Documentation ✔









https://poloclub.github.io/transformer-explainer/

https://github.com/poloclub/transformer-explainer

https://github.com/rasbt/LLMs-from-scratch

https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up

https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11


