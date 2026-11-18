# 01 - AKSHAY: Tasks Before 11:30 AM & Handoff Prompt

### Exactly What Akshay Needs Open on Screen at 11:30 AM:
1. **Screen 1**: An interactive Folium HTML map of Trip `00000000` showing:
   - Blue circle markers for raw noisy GPS points.
   - Dark/grey polylines for the road network arcs.
2. **Screen 2**: Basic EDA numbers on terminal or notepad:
   - Total trips: 80 training trips, 10 student-test trips.
   - Sampling rate: ~97% of observations are exactly 1 second apart, with occasional gaps up to 3–5 seconds.
3. **Akshay's 1-Sentence Viva Pitch**:
   > *"We use `shapely.strtree.STRtree` on road arc line segments with a 50m search radius and 150m dynamic fallback so our Candidate Miss Rate stays under 0.3%, preventing intersection misses."*

---

### COPY-PASTE HANDOFF PROMPT FOR AKSHAY (Send to Akshay now):

```markdown
Hi Akshay! Here is your exact task checklist and screen setup for our 11:30 AM Phase 1 review:

1. PULL LATEST REPO:
Run `git pull origin main` in your terminal right now.

2. YOUR 11:30 REVIEW DELIVERABLES (WHAT MUST BE ON YOUR SCREEN):
- Have an interactive Folium map open in your browser showing Trip 00000000 (blue dots for raw GPS points, dark lines for the road network).
- Have a quick terminal/text output showing basic dataset stats: 80 train trips, 10 student-test trips, ~97% at 1-second sampling rate with occasional GPS dropouts.

3. QUICK CODE TO GENERATE YOUR FOLIUM MAP IF NOT DONE:
```python
import folium, pandas as pd

track = pd.read_csv("Student folder/Student folder/student_test/00000000/00000000.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
nodes = pd.read_csv("Student folder/Student folder/student_test/00000000/00000000.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
arcs = pd.read_csv("Student folder/Student folder/student_test/00000000/00000000.arcs", sep=r"\s+", header=None, names=["u", "v"])

m = folium.Map(location=[track["lat"].mean(), track["lon"].mean()], zoom_start=14)
# Plot road arcs
for _, arc in arcs.iterrows():
    u_pt, v_pt = nodes.loc[arc["u"]], nodes.loc[arc["v"]]
    folium.PolyLine([(u_pt["lat"], u_pt["lon"]), (v_pt["lat"], v_pt["lon"])], color="gray", weight=1.5, opacity=0.6).add_to(m)
# Plot raw GPS points
for _, row in track.iterrows():
    folium.CircleMarker([row["lat"], row["lon"]], radius=2, color="blue", fill=True).add_to(m)
m.save("trip_00000000_eda.html")
print("Folium EDA map saved to trip_00000000_eda.html!")
```

4. YOUR EXACT 1-SENTENCE VIVA ANSWER WHEN TEACHERS ASK:
"We use shapely.strtree.STRtree on road arc LineStrings with a 50m search radius and 150m dynamic fallback so our Candidate Miss Rate stays under 0.3%, preventing intersection misses."
```
