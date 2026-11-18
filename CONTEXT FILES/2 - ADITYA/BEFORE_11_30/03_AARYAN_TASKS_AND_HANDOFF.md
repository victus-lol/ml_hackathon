# 03 - AARYAN: Tasks Before 11:30 AM & Handoff Prompt

### Exactly What Aaryan Needs Open on Screen at 11:30 AM:
1. **Laptop 1 Terminal**: Running FastAPI with Uvicorn:
   - `uvicorn --host 0.0.0.0 --port 8000`
   - Responding to `GET /api/v1/health`.
2. **Laptop 2 Browser**: Live connection to Laptop 1:
   - URL: `http://<laptop1_ip>:8000/api/v1/health`
   - Output displayed: `{"status": "healthy", "service": "PathSnap Model & HMM API"}`
3. **Repository Open**:
   - `https://github.com/victus-lol/ml_hackathon` open showing clean repository structure and active commit history.
4. **Aaryan's 1-Sentence Viva Pitch**:
   > *"Laptop 1 hosts the model and HMM inference service. Laptop 2 is the client UI rendering interactive Folium maps with strictly zero ground truth transmitted over the wire."*

---

### COPY-PASTE HANDOFF PROMPT FOR AARYAN (Send to Aaryan now):

```markdown
Hi Aaryan! Here is your exact task checklist and screen setup for our 11:30 AM Phase 1 review:

1. PULL LATEST REPO:
Run `git pull origin main` in your terminal right now.

2. YOUR 11:30 REVIEW DELIVERABLES (WHAT MUST BE ON YOUR SCREEN):
- Laptop 1 running FastAPI on port 8000 with a clean `/api/v1/health` endpoint.
- Laptop 2 connected over local Wi-Fi/Hotspot opening `http://<laptop1_ip>:8000/api/v1/health` to prove live 2-laptop communication to evaluators.
- GitHub repo open on screen showing clean folder layout and recent commits.

3. QUICK CODE TO RUN FASTAPI SKELETON ON LAPTOP 1 IF NOT DONE:
```python
# Save as server.py and run: uvicorn server:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI

app = FastAPI(title="PathSnap Model & HMM API")

@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "service": "PathSnap Model & HMM API",
        "wire_protocol": "Strict Zero Ground-Truth Isolation",
        "ready": True
    }

@app.get("/api/v1/trips")
def get_trips():
    return {"student_test_trips": ["00000000", "00000007", "00000009", "00000040", "00000052"]}
```

4. YOUR EXACT 1-SENTENCE VIVA ANSWER WHEN TEACHERS ASK:
"Laptop 1 hosts the model and HMM inference service. Laptop 2 is the client UI rendering interactive Folium maps with strictly zero ground truth transmitted over the wire."
```
