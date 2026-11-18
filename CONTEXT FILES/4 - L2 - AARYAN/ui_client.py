"""Laptop 2 Streamlit/Folium client; all trip data and predictions come from API."""

from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium


API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120


def fetch_trip_ids() -> list[str]:
    response = requests.get(f"{API_URL}/api/v1/trips", timeout=10)
    response.raise_for_status()
    return response.json().get("trip_ids", [])


def draw_result(result: dict) -> folium.Map:
    gps = result.get("track_coordinates", [])
    predicted = result.get("predicted_coordinates", [])
    if not gps:
        raise ValueError("API response did not include the raw GPS trace.")

    center = [gps[0][1], gps[0][0]]
    map_view = folium.Map(location=center, zoom_start=15, control_scale=True)
    gps_group = folium.FeatureGroup(name="Raw GPS", show=True)
    predicted_group = folium.FeatureGroup(name="Predicted route", show=True)
    folium.PolyLine(
        locations=[[point[1], point[0]] for point in gps],
        color="blue",
        weight=3,
        opacity=0.75,
        tooltip="Raw GPS trace",
    ).add_to(gps_group)
    for point in gps:
        folium.CircleMarker(
            location=[point[1], point[0]],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.65,
        ).add_to(gps_group)

    if predicted:
        folium.PolyLine(
            locations=[[point[1], point[0]] for point in predicted],
            color="red",
            weight=5,
            opacity=0.9,
            tooltip="Predicted route",
        ).add_to(predicted_group)
    gps_group.add_to(map_view)
    predicted_group.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    bounds = [[point[1], point[0]] for point in gps]
    map_view.fit_bounds(bounds)
    return map_view


st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap — route reconstruction")
st.caption("Choose a trip. The API loads the input and returns the prediction.")
st.caption(f"API: {API_URL}")

try:
    trip_ids = fetch_trip_ids()
except requests.RequestException as error:
    st.error(f"Could not reach the PathSnap API: {error}")
    st.stop()

if not trip_ids:
    st.info("The API has no trip IDs configured yet.")
    st.stop()

selected_trip = st.selectbox("Trip ID", trip_ids)
if st.button("Match trip", type="primary"):
    try:
        response = requests.post(
            f"{API_URL}/api/v1/match",
            json={"trip_id": selected_trip},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
        left, right = st.columns([3, 1])
        with left:
            st_folium(draw_result(result), height=620, use_container_width=True)
        with right:
            st.subheader("API result")
            st.write(f"Trip: {result.get('trip_id', selected_trip)}")
            st.metric("API latency", f"{result.get('latency_ms', 0):.2f} ms")
            for name, value in result.get("metrics", {}).items():
                st.metric(name.replace("_", " ").title(), value)
            st.write("Predicted arc count:", len(result.get("predicted_route", [])))
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"Could not display the API response: {error}")
