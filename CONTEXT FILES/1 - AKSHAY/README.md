# 1 - AKSHAY: Spatial Data Engineering & Baseline Models

### Primary Deliverables
1. **Raw Ingestion**: Read `.track`, `.nodes`, `.arcs`, and `.route` across all 80 train and 10 student-test trips.
2. **Metric Coordinate Projection**: Convert lat/lon into metric distances (meters) using local UTM or Haversine projection (never flat Euclidean degrees).
3. **Arc-Geometry Spatial Indexing**: Build `shapely.strtree.STRtree` over road arc line segments to query candidate road segments within 30m–50m (KD-Tree on nodes alone will miss long arcs).
4. **Feature Engineering**: Compute perpendicular distance, heading delta $|\theta_{\text{GPS}} - \theta_{\text{arc}}|$, velocity, segment length, and projection fraction.
5. **EDA & Visuals**: Missing-value analysis, GPS gap distributions, road network connectivity stats, and the initial Folium EDA map.
6. **Baseline Models (Post 11:30 AM)**:
   - **Model 1**: Regularized Logistic Regression.
   - **Model 2**: Distance-Weighted $k$-Nearest Neighbors ($k$-NN).
   - Write the 9-question report answers for Models 1 & 2.
