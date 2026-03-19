# LendSynthetix Dashboard — Setup Guide

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run app.py 
streamlit run app.py
```

Opens at: http://localhost:8501

---

## Sidebar Configuration

| Field | What to Enter |
|---|---|
| Langflow API Key | Your token from Langflow Cloud → Settings → API Keys |
| ARIA Flow ID | Flow ID from your ARIA agent flow URL |
| REX Flow ID | Flow ID from your REX agent flow URL |
| CLARA Flow ID | Flow ID from your CLARA agent flow URL |
| ORACLE Flow ID | Flow ID from your ORACLE merger flow URL |

**Finding your Flow ID:** Open your flow in Langflow Cloud.  
The URL will look like: `https://langflow.astra.datastax.com/flow/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`  
The long string at the end is your Flow ID.

---

## Demo Mode

Check ✅ **Demo Mode** in the sidebar to run with mock outputs — 
no API key needed. Perfect for testing the UI and your demo video.

---

## How the API Call Works

Each agent flow is called independently via:
```
POST https://api.langflow.astra.datastax.com/lf/{flow_id}/api/v1/run/chat
```

ORACLE receives all three agent outputs concatenated as its input,
plus the original application. This mirrors the debate structure.

---

## Connecting BioSentinel

The fraud score slider feeds directly into the application text 
sent to all agents. Both REX and CLARA are prompted to act on it.
When you build the real LSTM model, replace the slider with:

```python
fraud_score = biosential_model.predict(session_features)
```

---

## File Structure
```
lendsynthetix_dashboard/
├── app.py           ← Main dashboard (this file)
├── requirements.txt ← Dependencies
└── README.md        ← This file
```
