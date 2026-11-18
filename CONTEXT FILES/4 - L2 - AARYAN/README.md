# 4 - L2 - AARYAN: Distributed Deployment, Folium UI & Model 5

### Primary Deliverables
1. **GitHub Repository & Environment Setup**: Manage clean repository structure, `requirements.txt`, and installation instructions in `README.md`.
2. **Laptop 1 FastAPI Backend**:
   - `GET /api/v1/health`: Server health, model loaded, memory usage.
   - `GET /api/v1/trips`: List of available test trips.
   - `POST /api/v1/match`: Accepts raw arrays (`.track`, `.nodes`, `.arcs`) transmitted over the wire from Laptop 2, runs pipeline (Features -> ML -> HMM), and returns predicted route GeoJSON + metrics.
3. **Laptop 2 Safe Folium UI (Streamlit / Modern Web App)**:
   - **Evaluator Trips (CRITICAL RULE)**: Shows **ONLY** raw GPS observations (blue dots), PathSnap predicted route (red line), and server-computed metrics. **Zero ground-truth `.route` data may be transmitted or displayed.**
   - **Student-Test Trips (Offline)**: Optional diagnostic overlay for internal report screenshots.
4. **Model 5 (Neural Network / MLP)**:
   - Train and benchmark Multi-Layer Perceptron (MLP) on standardized 14-feature dataset.
   - Write 9-question report answers for Model 5.
5. **2-Laptop LAN / Hotspot Testing**: Verify connection (`uvicorn --host 0.0.0.0 --port 8000`), firewall whitelist, and $<85\text{ms}$ latency.
6. **Final Report Assembly**: Collate all 6 sections and export the consolidated PDF.
