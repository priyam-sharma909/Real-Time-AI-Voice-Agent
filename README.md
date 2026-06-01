# 🎙️ Real-Time AI Voice Agent

A real-time AI-powered voice agent built using **Pipecat**, **Sarvam AI**, and **Gemini/OpenAI LLMs** to conduct automated customer feedback surveys over phone calls.

The agent interacts with customers in natural conversations, supports both **English and Hindi**, collects structured feedback, calculates **NPS (Net Promoter Score)**, and handles survey flows intelligently in real time.

---

## 🚀 Features

* Real-time voice conversations
* AI-powered survey automation
* English and Hindi language support
* Automatic NPS collection and classification
* Structured customer feedback gathering
* Low-latency speech-to-text and text-to-speech pipeline
* Configurable conversation flow
* Built using modern voice AI infrastructure

---

## 🏗️ Tech Stack

### Voice Pipeline

* Pipecat
* WebRTC

### Speech Processing

* Sarvam AI STT (Speech-to-Text)
* Sarvam AI TTS (Text-to-Speech)

### Language Model

* Gemini
* OpenAI Compatible Models

### Backend

* Python 3.10+

---

## 📂 Project Structure

```text
Real-Time-AI-Voice-Agent/
│
├── main.py                # Main application entry point
├── pyproject.toml         # Project dependencies
├── .env                   # Environment variables (not committed)
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/priyam-sharma909/Real-Time-AI-Voice-Agent.git
cd Real-Time-AI-Voice-Agent
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / Mac:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install .
```

or

```bash
pip install -e .
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

Example:

```env
OPENAI_API_KEY=your_openai_key
SARVAM_API_KEY=your_sarvam_key
```

---

## ▶️ Running the Agent

```bash
python main.py
```

Once started, the voice agent will initialize the speech pipeline and begin handling customer interactions according to the configured survey workflow.

---
