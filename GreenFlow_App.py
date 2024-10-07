import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import moviepy.editor as mp
import csv
# import lapx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.request
from PIL import Image

# Enable URL download for the model
cfg_enable_url_download = True

# URL for the YOLO model
url = "https://archive.org/download/yolo-model-1/YOLO_Model%20%281%29.pt"
model_path = "models/YOLO_Model (1).pt"

st.set_page_config(
    page_title="GreenFlow",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a custom design to beautify the interface
st.markdown("""
    <style>
    body {
        background-color: #021407; 
        color: #f0f0f0;
    }
    .stApp {
        background-color: #021407;
        background-image: linear-gradient(-45deg, #001f10, #002a12, #021407, #001f10);
        background-size: 400% 400%;
        animation: gradientAnimation 10s ease infinite;
    }
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header {
        text-align: center;
        padding: 20px 0;
    }
    .header h1 {
        font-size: 60px;
        font-family: 'Poppins', sans-serif;
        color: #fcfcfc;
    }
    .header p {
        font-size: 30px;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #013311 !important; 
        color: white !important;
        font-size: 22px !important;
        padding: 15px 40px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2) !important;
        transition: background-color 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #001f0b !important; 
        color: #fff !important;
    }

    </style>
""", unsafe_allow_html=True)

# Page title and logo
st.markdown("""
    <div class="header">
        <img src="https://i.imgur.com/J0mk9eN.png" width=500 alt='GreenFlow'>
    </div>
""", unsafe_allow_html=True)

# Custom design for buttons and GreenFlow title
custom_style = """
    <style>
    .stButton button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-size: 48px; /* Increase font size of the button text */
        width: 100%; /* Ensure buttons are the same width */
        padding: 60px; /* Larger padding for size consistency */
        border-radius: 8px;
    }
    .centered-title {
        text-align: center;
        font-size: 48px; /* Large font size */
        font-weight: bold;
        color: #4CAF50; /* Green color */
    }
    .greenflow-title {
        font-family: 'Poppins', sans-serif;
        font-size: 50px !important; /* Increase font size */
        font-weight: bold;
        color: #4CAF50; /* Green color */
        text-align: center;
        margin-bottom: 100px;
    }
    </style>
"""

# Add the design to the page
st.markdown(custom_style, unsafe_allow_html=True)

# Handling the page state
if 'page' not in st.session_state:
    st.session_state.page = 'Visual Content Processing'

def switch_page(page):
    st.session_state.page = page


# Enhancements to the sidebar
st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #001f0b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.5);
    }
    
    .sidebar .sidebar-buttons {
        margin-top: 20px;
    }

    .sidebar button {
        background-color: #013311 !important;
        color: white !important;
        font-size: 2000px !important; /* Increased font size for button text */
        padding: 15px 25px !important;
        margin-bottom: 10px;
        border-radius: 10px !important;
        width: 100% !important;
        transition: background-color 0.3s ease !important;
    }

    .sidebar button:hover {
        background-color: #002a12 !important;
    }

    .sidebar .objectives {
        margin-top: 30px;
        background-color: #013311;
        padding: 15px;
        border-radius: 10px;
    }

    .sidebar ul {
        padding-left: 20px;
    }

    .sidebar ul li {
        font-family: 'Arial', sans-serif;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .sidebar ul li strong {
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Enhancements to the sidebar
st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)

# GreenFlow system title with the new design
st.sidebar.markdown("<h3 class='greenflow-title'>GreenFlow</h3>", unsafe_allow_html=True)

# Adding control buttons to the interface
st.sidebar.markdown("<div class='sidebar-buttons'>", unsafe_allow_html=True)

if st.sidebar.button("Visual Content Processing"):
    st.session_state.page = "Visual Content Processing"
if st.sidebar.button("Dashboard"):
    st.session_state.page = "Dashboard"
if st.sidebar.button("SUMO Simulation"):
    st.session_state.page = "SUMO Simulation"
if st.sidebar.button("SUMO Simulation With Agent"):
    st.session_state.page = "SUMO Simulation With Agent"


st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)

#st.sidebar.image("https://www.vhv.rs/dpng/d/525-5255156_traffic-light-transparent-images-png-transparent-traffic-light.png", width=170)

img = Image.open("videos/traffic_light.png")

# عرض الصورة
st.sidebar.image(img, use_column_width=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

if st.session_state.page == "Visual Content Processing":
    st.markdown("""<div class="header"><h1> AI-Based Traffic Light</h1><p> Track, monitor, and manage traffic congestion in Riyadh city 🚦🚗</p></div>""", unsafe_allow_html=True)
    upload_file = st.file_uploader("Upload a Video", type=["avi", "mov"])

    if upload_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
        tfile.write(upload_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        t_outputfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
        output_video_path = t_outputfile.name

        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        # Define the ROIs, their counters, and corresponding colors
        rois = [
            (210, 610, 17, 235),  # ROI 1
            (235, 610, 17, 235),  # ROI 2
            (260, 610, 17, 235),  # ROI 3
            (285, 610, 17, 235),  # ROI 4
            (310, 610, 17, 235),  # ROI 5
            (130, 3, 17, 235),    # ROI 6
            (155, 3, 17, 235),    # ROI 7
            (180, 3, 17, 235),    # ROI 8
            (205, 3, 17, 235),    # ROI 9
            (230, 3, 17, 235)     # ROI 10
        ]

        roi_colors = [
            (255, 0, 0),    # Red for ROI 1
            (0, 255, 0),    # Green for ROI 2
            (0, 0, 255),    # Blue for ROI 3
            (255, 255, 0),  # Cyan for ROI 4
            (255, 0, 255),  # Magenta for ROI 5
            (0, 255, 255),  # Yellow for ROI 6
            (100, 255, 255),# Light Blue for ROI 7
            (255, 100, 0),  # Orange for ROI 8
            (100, 0, 255),  # Purple for ROI 9
            (0, 100, 255)   # Light Orange for ROI 10
        ]

        # Initialize vehicle count and tracked IDs for each ROI
        vehicle_count = [0] * len(rois)
        tracked_ids = [[] for _ in range(len(rois))]

        # To count unique detected vehicles regardless of ROI
        unique_vehicle_ids = set()  # Using a set to store unique vehicle IDs

        # Total vehicle counts for top and bottom streets
        top_street_count = 0
        bottom_street_count = 0

        frame_number = 0
        object_counts = []

        # Open CSV file for writing vehicle counts
        csv_file = 'vehicle_count_with_totals.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ROI', 'Vehicle Count'])  # CSV header

        # Main processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection
            results = model.track(frame, persist=True)

            for result in results:
                for obj in result.boxes:
                    bbox = obj.xyxy[0].cpu().numpy()  # Bounding box coordinates
                    obj_id = int(obj.id[0].cpu().numpy()) if obj.id is not None else -1  # Unique ID

                    # Add to unique vehicle IDs set
                    unique_vehicle_ids.add(obj_id)

                    x1, y1, x2, y2 = map(int, bbox)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Check each ROI for vehicle presence
                    for i, (roi_x, roi_y, roi_w, roi_h) in enumerate(rois):
                        if roi_x <= center_x <= roi_x + roi_w and roi_y <= center_y <= roi_y + roi_h:
                            # Check if the vehicle has already been counted in this ROI
                            if obj_id not in tracked_ids[i]:
                                # Increment the count for this ROI if the vehicle is within the ROI and not counted
                                vehicle_count[i] += 1
                                tracked_ids[i].append(obj_id)

                                # Update total counts based on ROI position
                                if i < 5:  # ROIs 1-5 (Top Street)
                                    top_street_count += 1
                                else:  # ROIs 6-10 (Bottom Street)
                                    bottom_street_count += 1

                    # Draw bounding box and label for each detected object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'ID: {obj_id}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw ROIs and display vehicle counts
            # for i, (roi_x, roi_y, roi_w, roi_h) in enumerate(rois):
            #     # Draw the ROI rectangle with its unique color
            #     roi_color = roi_colors[i]
            #     cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), roi_color, 2)

            # Display the vehicle count for each ROI in the top-left corner
            # y_offset = 30  # Initial y offset for displaying the count
            # for i, count in enumerate(vehicle_count):
            #     count_label = f'ROI {i + 1}: {count}'
            #     color = roi_colors[i]
            #     cv2.putText(frame, count_label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            #     y_offset += 30  # Move down for the next ROI count

            # Save the current frame to the output video
            out.write(frame)

        # Save the vehicle counts to the CSV file at the end of the video processing
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i, count in enumerate(vehicle_count):
                writer.writerow([f'ROI {i + 1}', count])
            # Add totals for the counts
            writer.writerow(['Total Top Street (ROIs 1-5)', top_street_count])
            writer.writerow(['Total Bottom Street (ROIs 6-10)', bottom_street_count])

        # Release resources
        cap.release()
        out.release()

        # Convert the video to MP4
        if os.path.exists(output_video_path):
            mp4_output_path = output_video_path.replace('.avi', '.mp4')
            clip = mp.VideoFileClip(output_video_path)
            clip.write_videofile(mp4_output_path)

            st.video(mp4_output_path)

            # Display the total vehicle count
            total_vehicles = len(unique_vehicle_ids)  # Count of unique vehicles detected
            st.markdown(f"<h3 style='color: white;'>Total detected vehicles: {total_vehicles}</h3>", unsafe_allow_html=True)

            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download Resulted Video",
                    data=video_file,
                    file_name="output_with_predictions.avi",
                    mime="video/avi"
                )
        else:
            st.error("Error: Video not saved correctly.")

if 'view' not in st.session_state:
    st.session_state.view = 'Before'

def set_view(view):
    st.session_state.view = view


if st.session_state.page == "Dashboard":
    st.markdown("""<div class="header"><h1>Traffic Analysis Dashboard</h1></div>""", unsafe_allow_html=True)

    # Load data
    df = pd.read_csv('data/vehicle_count_time.csv')

    # Dropdown for street selection
    selected_street = st.selectbox("Select a Road", ['Road 1', 'Road 2'])

    # Filter data based on selected street
    if selected_street == 'Road 1':
        df_filtered = df[df['ROI'].isin(['ROI 1', 'ROI 2', 'ROI 3', 'ROI 4', 'ROI 5'])]
    else:
        df_filtered = df[df['ROI'].isin(['ROI 6', 'ROI 7', 'ROI 8', 'ROI 9', 'ROI 10'])]

    # Calculate total and road-specific wait times
    total_wait_time = df['Average Wait Time (s)'].sum()
    road_wait_time = df_filtered['Average Wait Time (s)'].sum()
    percentage_wait_time = (road_wait_time / total_wait_time) * 100

    # Create three main columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])

    # Column 1: Busiest Lane Indicator and Average Wait Time Percentage
    with col1:
        st.markdown("<h3 style='text-align: center;'>Busiest Lane Indicator</h3>", unsafe_allow_html=True)
        
        # Identify busiest lane
        busiest_lane = df_filtered.loc[df_filtered['Vehicle Count'].idxmax()]
        fig_circle = go.Figure(go.Indicator(
            mode="gauge+number",
            value=busiest_lane['Vehicle Count'],
            title={'text': f"Busiest Lane: {busiest_lane['ROI']}"},
            gauge={'axis': {'range': [0, df_filtered['Vehicle Count'].max()]}, 'bar': {'color': "red"}}
        ))
        # Make the background transparent
        fig_circle.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_circle)
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

        # Display average wait time percentage
        st.markdown("<h3 style='text-align: center;'>Average Wait Time Percentage</h3>", unsafe_allow_html=True)
        fig_percentage = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage_wait_time,
            number={'suffix': "%"},
            title={'text': f'Percentage of Wait Time in {selected_street}'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 100], 'color': "gray"}]
            }
        ))
        # Make the background transparent
        fig_percentage.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_percentage)

    # Column 2: Pie Charts for Vehicle Count and Average Wait Time
    with col2:
        # Pie Chart for vehicle count
        st.markdown("<h3 style='text-align: center;'>Vehicle Count per Lane</h3>", unsafe_allow_html=True)
        colors = ['#e0f2e9', '#b2e0d6', '#80c5b5', '#4fb99a', '#26a68a']
        
        fig_pie_vehicles = px.pie(df_filtered, values='Vehicle Count', names='ROI', title=' ')
        fig_pie_vehicles.update_traces(marker=dict(colors=colors))
        fig_pie_vehicles.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            legend=dict(x=0.5, y=-0.1, xanchor="center", yanchor="top", orientation="h")
        )
        fig_pie_vehicles.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie_vehicles, use_container_width=True)
        st.markdown("<div style='margin-bottom: 60px;'></div>", unsafe_allow_html=True)

        # Pie Chart for average wait time
        st.markdown("<h3 style='text-align: center;'>Average Waiting Time per Lane</h3>", unsafe_allow_html=True)
        fig_pie_wait_time = px.pie(df_filtered, values='Average Wait Time (s)', names='ROI', title=' ')
        fig_pie_wait_time.update_traces(marker=dict(colors=colors))
        fig_pie_wait_time.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            legend=dict(x=0.5, y=-0.1, xanchor="center", yanchor="top", orientation="h")
        )
        fig_pie_wait_time.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie_wait_time, use_container_width=True)

    # Column 3: Congestion Levels with Progress Bars
    with col3:
        st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True) 
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Congestion Level</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)

        for index, row in df_filtered.iterrows():
            st.text(f"Lane {row['ROI']} - Vehicles: {row['Vehicle Count']}")
            congestion_level = min(row['Vehicle Count'] / 50, 1.0)  # Normalize vehicle count
            
            # Determine congestion level color
            if congestion_level >= 0.7:  # High congestion
                inner_color = "#ff3333"  # Red
            elif congestion_level >= 0.4:  # Medium congestion
                inner_color = "#ffcc00"  # Orange
            else:  # Low congestion
                inner_color = "#00b300"  # Green

            # Progress bar
            st.markdown(f"""
                <div style="position: relative; height: 15px; background-color: #2d2d2d; border-radius: 10px;">
                    <div style="width: {congestion_level * 100}%; height: 100%; background-color: {inner_color}; border-radius: 10px;"></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)

        # Bar Chart for total vehicles and average wait time
        st.markdown(f"<h3 style='text-align: center;'>Total Vehicles and Average Wait Time for {selected_street}</h3>", unsafe_allow_html=True)

        # Create summary DataFrame for total vehicles and average wait time
        df['Street'] = df['ROI'].apply(lambda x: 'Road 1' if x in ['ROI 1', 'ROI 2', 'ROI 3', 'ROI 4', 'ROI 5'] else ('Road 2' if x in ['ROI 6', 'ROI 7', 'ROI 8', 'ROI 9', 'ROI 10'] else 'Other'))
        summary_df = df[df['Street'] != 'Other'].groupby('Street').agg(
            Total_Vehicles=('Vehicle Count', 'sum'),
            Average_Wait_Time=('Average Wait Time (s)', 'mean')
        ).reset_index()

        # Filter summary DataFrame based on selected street
        filtered_summary_df = summary_df[summary_df['Street'] == selected_street]

        # Create bar chart
        fig_bar = px.bar(filtered_summary_df, x='Street', 
                         y=['Total_Vehicles', 'Average_Wait_Time'], 
                         barmode='group', 
                         title=' ',
                         labels={'value': 'Count', 'Street': 'Street'},
                         color_discrete_sequence=['#a8ddb5', '#41ab5d'])  # Change colors as needed
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

if st.session_state.page == "SUMO Simulation":
    st.markdown("""<div class="header"><h1> Traffic Analysis SUMO Simulation </h1></div>""", unsafe_allow_html=True)

    # Load data from your latest file
    df = pd.read_csv('data/final_lane_road_data2.csv')

    sumo_video_path2 = "videos/After_Agent.mp4"
    st.video(sumo_video_path2)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    selected_street = st.selectbox("Select a Road", ['Road 1', 'Road 2', 'Road 3', 'Road 4'])

    # Filter data based on the selected street
    if selected_street == 'Road 1':
        df_filtered = df[df['Edge ID'] == '636647587#2'][df['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]
    elif selected_street == 'Road 2':
        df_filtered = df[df['Edge ID'] == '1306997822#2'][df['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]
    elif selected_street == 'Road 3':
        df_filtered = df[df['Edge ID'] == '159072600#3'][df['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]
    else:
        df_filtered = df[df['Edge ID'] == '53823318#1'][df['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]

    # Calculate the percentage of average wait time
    total_wait_time = df['Average Waiting Time (Lane) (s)'].sum()
    road_wait_time = df_filtered['Average Waiting Time (Lane) (s)'].sum()
    percentage_wait_time = (road_wait_time / total_wait_time) * 100

    # Split the page into 3 main columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the first column to display congestion indicators and percentage
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>Busiest Lane Indicator</h3>", unsafe_allow_html=True)
        
        # Calculate the busiest lane
        busiest_lane = df_filtered.loc[df_filtered['Total Vehicle Count'].idxmax()]
        fig_circle = go.Figure(go.Indicator(
            mode="gauge+number",
            value=busiest_lane['Total Vehicle Count'],
            title={'text': f"Busiest Lane: {busiest_lane['Lane ID']}"},
            gauge={
                'axis': {'range': [0, df_filtered['Total Vehicle Count'].max()]},
                'bar': {'color': "red"}
            }
        ))
        # Adjust the background to make it transparent
        fig_circle.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Remove paper background
            plot_bgcolor='rgba(0,0,0,0)'    # Remove plot background
        )
        st.plotly_chart(fig_circle)

        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

        # Display average wait time percentage
        st.markdown(f"<h3 style='text-align: center;'>Average Wait Time Percentage</h3>", unsafe_allow_html=True)
        
        fig_percentage = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage_wait_time,
            number={'suffix': "%"},
            title={'text': f'Percentage of Wait Time in {selected_street}'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}]
            }
        ))
        # Adjust the background to make it transparent
        fig_percentage.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Remove paper background
            plot_bgcolor='rgba(0,0,0,0)'    # Remove plot background
        )
        st.plotly_chart(fig_percentage)

    # Use the second column to display charts (Pie Charts)
    with col2:
        # Pie chart for vehicle counts
        st.markdown(f"<h3 style='text-align: center;'>Vehicle Count per Lane</h3>", unsafe_allow_html=True)
        colors = ['#e0f2e9', '#b2e0d6', '#4fb99a']

        fig_pie_vehicles = px.pie(df_filtered, values='Total Vehicle Count', names='Lane ID', title=' ')
        fig_pie_vehicles.update_traces(marker=dict(colors=colors))
        fig_pie_vehicles.update_layout(
            title={
                'x': 0.5,  # Center title horizontally
                'xanchor': 'center',  # Ensure center is the anchoring point
                'yanchor': 'top'  # Title at the top
            },
            legend=dict(
                x=0.5,  # Center legend horizontally
                y=-0.1,  # Legend at the bottom vertically, adjust as needed
                xanchor="center",  # Center legend
                yanchor="top",  # Align legend vertically at the top
                orientation="h"  # Horizontal legend
            )
        )
        # Adjust the background to make it transparent
        fig_pie_vehicles.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Remove paper background
            plot_bgcolor='rgba(0,0,0,0)'    # Remove plot background
        )
        st.plotly_chart(fig_pie_vehicles, use_container_width=True)
        
        st.markdown("<div style='margin-bottom: 60px;'></div>", unsafe_allow_html=True)

        # Pie chart for average wait time
        st.markdown(f"<h3 style='text-align: center;'>Average Waiting Time per Lane</h3>", unsafe_allow_html=True)

        fig_pie_wait_time = px.pie(df_filtered, values='Average Waiting Time (Lane) (s)', names='Lane ID', title=' ')
        fig_pie_wait_time.update_traces(marker=dict(colors=colors))
        fig_pie_wait_time.update_layout(
            title={
                'x': 0.5,  # Center title horizontally
                'xanchor': 'center',  # Ensure center is the anchoring point
                'yanchor': 'top'  # Title at the top
            },
            legend=dict(
                x=0.5,  # Center legend horizontally
                y=-0.1,  # Legend at the bottom vertically, adjust as needed
                xanchor="center",  # Center legend
                yanchor="top",  # Align legend vertically at the top
                orientation="h"  # Horizontal legend
            )
        )
        # Adjust the background to make it transparent
        fig_pie_wait_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Remove paper background
            plot_bgcolor='rgba(0,0,0,0)'    # Remove plot background
        )
        st.plotly_chart(fig_pie_wait_time, use_container_width=True)

    # Use the third column to display congestion status using Progress Bars
    with col3:
        st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True) 
        st.markdown(f"<h3 style='text-align: center; margin-bottom: 20px;'>Congestion Level</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)

        for index, row in df_filtered.iterrows():
            st.text(f"Lane {row['Lane ID']} - Vehicles: {row['Total Vehicle Count']}")
            congestion_level = min(row['Total Vehicle Count'] / 50, 1.0)  # Normalize vehicle count
            
            # Determine congestion level color
            if congestion_level >= 0.7:  # High congestion
                inner_color = "#ff3333"  # Red
            elif congestion_level >= 0.4:  # Medium congestion
                inner_color = "#ffcc00"  # Orange
            else:  # Low congestion
                inner_color = "#00b300"  # Green

            # Create a progress bar with a fixed outer color and a variable inner color
            st.markdown(f"""
                <div style="position: relative; height: 15px; background-color: #2d2d2d; border-radius: 10px;">
                    <div style="width: {congestion_level * 100}%; height: 100%; background-color: {inner_color}; border-radius: 10px;"></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 250px;'></div>", unsafe_allow_html=True)

        # Bar chart to show total vehicles and average wait time for each street below Progress Bars
        st.markdown(f"<h3 style='text-align: center;'>Total Vehicles and Average Wait Time for {selected_street}</h3>", unsafe_allow_html=True)

        # Create a bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_filtered['Lane ID'],
            y=df_filtered['Total Vehicle Count'],
            name='Total Vehicles',
            marker_color='lightskyblue'
        ))
        fig_bar.add_trace(go.Bar(
            x=df_filtered['Lane ID'],
            y=df_filtered['Average Waiting Time (Lane) (s)'],
            name='Average Wait Time (s)',
            marker_color='orange'
        ))

        # Customize layout
        fig_bar.update_layout(
            barmode='group',
            title_text=f'Total Vehicles and Average Wait Time for {selected_street}',
            xaxis_title='Lane ID',
            yaxis_title='Count / Time (s)',
            legend=dict(x=0.5, y=-0.1, xanchor='center', yanchor='top', orientation='h'),
            paper_bgcolor='rgba(0,0,0,0)',  # Remove paper background
            plot_bgcolor='rgba(0,0,0,0)'    # Remove plot background
        )
        st.plotly_chart(fig_bar)

if st.session_state.page == "SUMO Simulation With Agent":
    st.markdown("""<div class="header"><h1> SUMO Simulation Dashboard (With Agent)</h1></div>""", unsafe_allow_html=True)

    # Load data after applying the Agent
    df_with_agent = pd.read_csv('data/simulation_data_last.csv')
    sumo_video_path = "videos/Befor_Agent.mp4"
    st.video(sumo_video_path)
    st.markdown("<h1 style='font-size: 36px;'>This video shows the simulated traffic signal actions using SUMO</h1>", unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    # Select the street
    selected_street = st.selectbox("Select a Road", ['Road 1', 'Road 2', 'Road 3', 'Road 4'])

    # Filter data based on the selected street
    if selected_street == 'Road 1':
        df_with_agent_filtered = df_with_agent[df_with_agent['Edge ID'] == '636647587#2'][df_with_agent['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]
    elif selected_street == 'Road 2':
        df_with_agent_filtered = df_with_agent[df_with_agent['Edge ID'] == '1306997822#2'][df_with_agent['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]
    elif selected_street == 'Road 3':
        df_with_agent_filtered = df_with_agent[df_with_agent['Edge ID'] == '159072600#3'][df_with_agent['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]
    else:
        df_with_agent_filtered = df_with_agent[df_with_agent['Edge ID'] == '53823318#1'][df_with_agent['Lane ID'].isin(['Lane 1', 'Lane 2', 'Lane 3'])]

    # Calculate the percentage of average wait time
    total_wait_time = df_with_agent['avg_waiting_time'].sum()
    road_wait_time = df_with_agent_filtered['avg_waiting_time'].sum()
    percentage_wait_time = (road_wait_time / total_wait_time) * 100 if total_wait_time != 0 else 0

    # Divide the page into 3 main columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the first column to display congestion indicators and percentage
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>Busiest Lane Indicator</h3>", unsafe_allow_html=True)

        # Calculate the busiest lane
        busiest_lane = df_with_agent_filtered.loc[df_with_agent_filtered['Total Vehicle Count'].idxmax()]
        fig_circle = go.Figure(go.Indicator(
            mode="gauge+number",
            value=busiest_lane['Total Vehicle Count'],
            title={'text': f"Busiest Lane: {busiest_lane['Lane ID']}"},
            gauge={
                'axis': {'range': [0, df_with_agent_filtered['Total Vehicle Count'].max()]},
                'bar': {'color': "red"}
            }
        ))
        fig_circle.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_circle)
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

        # Display the average wait time percentage
        st.markdown(f"<h3 style='text-align: center;'>Average Wait Time Percentage</h3>", unsafe_allow_html=True)
        fig_percentage = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage_wait_time,
            number={'suffix': "%"},
            title={'text': f'Percentage of Wait Time in {selected_street}'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}]
            }
        ))
        fig_percentage.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_percentage)

    # Use the second column to display pie charts
    with col2:
        # Pie chart for vehicle count
        st.markdown(f"<h3 style='text-align: center;'>Vehicle Count per Lane</h3>", unsafe_allow_html=True)
        colors = ['#e0f2e9', '#b2e0d6','#4fb99a']

        fig_pie_vehicles = px.pie(df_with_agent_filtered, values='Total Vehicle Count', names='Lane ID', title=f' ')
        fig_pie_vehicles.update_traces(marker=dict(colors=colors))
        fig_pie_vehicles.update_layout(
            title={
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Ensure the center is the anchor point
                'yanchor': 'top'  # Title at the top
            },
            legend=dict(
                x=0.5,  # Center the legend horizontally
                y=-0.1,  # Legend at the bottom vertically; adjust as needed
                xanchor="center",  # Center the legend
                yanchor="top",  # Vertically align to the top
                orientation="h"  # Horizontal legend
            )
        )
        fig_pie_vehicles.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie_vehicles, use_container_width=True)
        st.markdown("<div style='margin-bottom: 60px;'></div>", unsafe_allow_html=True)

        # Pie chart for average wait time
        st.markdown(f"<h3 style='text-align: center;'>Average Waiting Time per Lane</h3>", unsafe_allow_html=True)

        fig_pie_wait_time = px.pie(df_with_agent_filtered, values='avg_waiting_time', names='Lane ID', title=f' ')
        fig_pie_wait_time.update_traces(marker=dict(colors=colors))
        fig_pie_wait_time.update_layout(
            title={
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Ensure the center is the anchor point
                'yanchor': 'top'  # Title at the top
            },
            legend=dict(
                x=0.5,  # Center the legend horizontally
                y=-0.1,  # Legend at the bottom vertically; adjust as needed
                xanchor="center",  # Center the legend
                yanchor="top",  # Vertically align to the top
                orientation="h"  # Horizontal legend
            )
        )
        
        fig_pie_wait_time.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title_x=0.5)
        st.plotly_chart(fig_pie_wait_time, use_container_width=True)

    # Use the third column to display congestion status using progress bars
    with col3:
            st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True) 
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 20px;'>Congestion Level</h3>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)

            for index, row in df_with_agent_filtered.iterrows():
                st.text(f"Lane {row['Lane ID']} - Vehicles: {row['Total Vehicle Count']}")

                congestion_level = min(row['Total Vehicle Count'] / 50, 1.0)  # Normalize vehicle count

                # Apply color conditions based on congestion level
                if congestion_level >= 0.7:  # High congestion
                    inner_color = "#ff3333"  # Red inner color
                elif congestion_level >= 0.4:  # Medium congestion
                    inner_color = "#ffcc00"  # Orange inner color
                else:  # Low congestion
                    inner_color = "#00b300"  # Green inner color

                # Apply progress bar with fixed outer color and variable inner color
                st.markdown(f"""
                <div style="position: relative; height: 15px; background-color: #2d2d2d; border-radius: 10px;">
                    <div style="width: {congestion_level * 100}%; height: 100%; background-color: {inner_color}; border-radius: 10px;"></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom: 250px;'></div>", unsafe_allow_html=True)

            # Bar chart to illustrate the total number of vehicles and average wait time for each street under progress bars
            st.markdown(f"<h3 style='text-align: center;'>Total Vehicles and Average Wait Time for {selected_street}</h3>", unsafe_allow_html=True)

            df['Street'] = df['Edge ID'].apply(lambda x: 'Road 1' if x == '636647587#2' else 
                                                     'Road 2' if x == '1306997822#2' else 
                                                     'Road 3' if x == '159072600#3' else 
                                                     'Road 4')
    
            # Filter the DataFrame to include only rows where 'Lane ID' is 'All Lanes'
            df_all_lanes = df[df['Lane ID'] == 'All Lanes']
            
            # Group by the new 'Street' column and calculate the total vehicles and average wait time
            summary_df = df_all_lanes.groupby('Street').agg(
                            Total_Vehicles=('Total Vehicle Count', 'sum'),
                            Average_Wait_Time=('Average Waiting Time (Road) (s)', 'mean')
                        ).reset_index()

            # Draw a bar chart in the same column
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=df_street_summary_all_lanes['Street'], y=df_street_summary_all_lanes['Total Vehicle Count'],
                                      name='Total Vehicle Count', marker_color='blue'))
            fig_bar.add_trace(go.Bar(x=df_street_summary_all_lanes['Street'], y=df_street_summary_all_lanes['avg_waiting_time'],
                                      name='Average Waiting Time', marker_color='orange'))
            fig_bar.update_layout(barmode='group', title=f'Total Vehicle Count and Average Wait Time for {selected_street}')
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
