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
# import lap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.request

# Enable URL download for the model
cfg_enable_url_download = True

# URL for the YOLO model
url = "https://archive.org/download/yolo-model-1/YOLO_Model%20%281%29.pt"
model_path = "models/YOLO_Model (1).pt"

# Ensure the model directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Download the model if it doesn't exist
if cfg_enable_url_download and not os.path.exists(model_path):
    urllib.request.urlretrieve(url, model_path)

# Load the YOLO model
model = YOLO(model_path)

# Streamlit page configuration
st.set_page_config(
    page_title="GreenFlow",
    page_icon="🚦",
    layout="centered"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #333333;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #0a2112, #56635a);
        background-size: cover;
    }
    .header {
        text-align: center;
        padding: 20px 0;
    }
    .header h1 {
        font-family: 'Arial', sans-serif;
        color: #fcfcfc;
    }
    .header p {
        font-size: 18px;
        color: #1b1c1b;
    }
    .upload-section {
        text-align: center;
    }
    .stButton > button {
        background-color: #81cc81 !important; 
        color: white !important;
        font-size: 22px !important;
        padding: 15px 40px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2) !important;
        transition: background-color 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #228B22 !important; 
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center;'>
        <img src="https://i.imgur.com/J0mk9eN.png" width=250 alt='GreenFlow'>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Control Panel")
page = st.sidebar.selectbox("Choose a page", ["Visual Content Processing", "Dashboard", "SUMO Simulation", "SUMO Simulation With Agent"])

if page == "Visual Content Processing":
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

# Load the CSV data
df = pd.read_csv('data/vehicle_count_time.csv')

# Assuming your dataframe is named df and already available
# Rename columns to replace 'ROI' with 'Lane'
df.rename(columns={'ROI': 'Lane'}, inplace=True)

# Create a green color palette
green_palette = ['#e0f2e9', '#b2e0d6', '#80c5b5', '#4fb99a', '#26a68a']

if page == "Dashboard":


    st.markdown("""<div class="header"><h1> Traffic Analysis Dashboard </h1></div>""", unsafe_allow_html=True)

    # Create subplots for Road 1 and Road 2 pie charts (Vehicle Count)
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=["Road 1 - Vehicle Count", "Road 2 - Vehicle Count"],
        specs=[[{'type': 'domain'}, {'type': 'domain'}]]  # Domain type for pie charts
    )

    # Pie chart for Vehicle Count by Lane for Road 1 and Road 2
    for i in range(2):
        road_data = df.iloc[i * 5:(i + 1) * 5]  # Slicing for each road

        fig.add_trace(go.Pie(
            labels=road_data['Lane'],
            values=road_data['Vehicle Count'],
            marker=dict(colors=green_palette),  # Apply green color palette
            textinfo='label+value',
            hoverinfo='label+value+percent',
            textposition='inside',
            hole=0.4
        ), row=1, col=i+1)

    # Update layout to adjust the titles and remove legend
    fig.update_layout(
        title_text="Vehicle Count Distribution by Road",
        title_x=0.25,  # Move the title slightly to the right
        title_font=dict(size=20, family='Arial', color='black', weight='normal'),
        annotations=[dict(text='Road 1', x=0.23, y=0.470, font_size=20, showarrow=False),
                    dict(text='Road 2', x=0.78, y=0.47, font_size=20, showarrow=False)],
        showlegend=False
    )

    st.plotly_chart(fig)

    # Create another figure for Average Wait Time pie charts
    fig2 = make_subplots(
        rows=1, cols=2, 
        subplot_titles=["Road 1 - Average Waiting Time", "Road 2 - Average Wait Time"],
        specs=[[{'type': 'domain'}, {'type': 'domain'}]]  # Domain type for pie charts
    )

    # Pie chart for Average Wait Time by Lane for Road 1 and Road 2
    for i in range(2):
        road_data = df.iloc[i * 5:(i + 1) * 5]  # Slicing for each road

        fig2.add_trace(go.Pie(
            labels=road_data['Lane'],
            values=road_data['Average Wait Time (s)'],
            marker=dict(colors=green_palette),  # Apply green color palette
            textinfo='label+value',
            hoverinfo='label+value+percent',
            textposition='inside',
            hole=0.4
        ), row=1, col=i+1)

    # Update layout to adjust the titles and remove legend
    fig2.update_layout(
        title_text="Average Waiting Time Distribution by Road",
        title_x=0.2,  # Move the title slightly to the right
        title_font=dict(size=20, family='Arial', color='black', weight='normal'),
        annotations=[dict(text='Road 1', x=0.23, y=0.470, font_size=20, showarrow=False),
                    dict(text='Road 2', x=0.78, y=0.47, font_size=20, showarrow=False)],
        showlegend=False
    )

    st.plotly_chart(fig2)

    # Bar chart for total vehicles and average wait time
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = ['Road 1', 'Road 2']
    vehicle_counts = [df.iloc[0:5]['Vehicle Count'].sum(), df.iloc[5:10]['Vehicle Count'].sum()]
    average_wait_times = [df.iloc[0:5]['Average Wait Time (s)'].mean(), df.iloc[5:10]['Average Wait Time (s)'].mean()]

    # Create clustered bar chart with green colors
    bar1 = ax3.bar(index, vehicle_counts, bar_width, label='Total Vehicles', color=green_palette[2], alpha=0.7)
    bar2 = ax3.bar(index, average_wait_times, bar_width, label='Average Waiting Time (s)', color=green_palette[4], alpha=0.7, bottom=vehicle_counts)

    ax3.set_ylabel('Count')
    ax3.set_title('Total Vehicles and Average Waiting Time by Road')  # Adding title to the bar chart
    ax3.legend()
    ax3.set_xticks(index)
    ax3.set_xticklabels(index)

    # Add numbers inside the bars
    for bars in [bar1, bar2]:
        for bar in bars:
            yval = bar.get_height() + (bar.get_y() if bar in bar1 else 0)  # Adjust y for stacked bars
            ax3.text(bar.get_x() + bar.get_width() / 2, yval, round(bar.get_height(), 2), ha='center', va='bottom')

    st.pyplot(fig3)

df1 = pd.read_csv('data/final_lane_road_data2.csv')
if page == "SUMO Simulation":

    st.markdown("""<div class="header"><h1> SUMO Simulation Dashboard </h1></div>""", unsafe_allow_html=True)
    sumo_video_path = "videos/Befor_Agent.mp4"
    st.video(sumo_video_path)
    # st.write("This video shows the simulated traffic signal actions using SUMO")


    # Create a green color palette
    green_palette = ['#e0f2e9', '#b2e0d6', '#80c5b5', '#4fb99a', '#26a68a']  # Light to dark green
    
    # Create a green color palette
    green_palette = ['#e0f2e9', '#b2e0d6', '#80c5b5', '#4fb99a', '#26a68a']  # Light to dark green

    # Subplots for four roads
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Road 1 - Vehicle Count", "Road 2 - Vehicle Count", "Road 3 - Vehicle Count", "Road 4 - Vehicle Count"],
        specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]  # Domain type for pie charts
    )

    # Create pie charts for Vehicle Count by Lane for each Road
    for i in range(4):
        road_data = df1[(df1['Edge ID'] == df1['Edge ID'].unique()[i]) & (df1['Lane ID'] != 'All Lanes')]

        fig.add_trace(go.Pie(
            labels=road_data['Lane ID'],
            values=road_data['Total Vehicle Count'],
            marker=dict(colors=green_palette),  # Apply green color palette
            textinfo='label+value',
            hoverinfo='label+value+percent',
            textposition='inside',
            hole=0.4
        ), row=(i // 2) + 1, col=(i % 2) + 1)

    # Update layout to adjust the titles and remove legend
    # Update layout to add custom title for vehicle count distribution
    fig.update_layout(
        title_text="Vehicle Count Distribution by Road",
        title_x=0.25,  # Adjust the title position
        title_font=dict(size=20, family='Arial', color='black', weight='normal'),
        annotations=[dict(text='Road 1', x=0.1, y=0.8, font_size=20, showarrow=False),
                    dict(text='Road 2', x=0.63, y=0.8, font_size=20, showarrow=False),
                    dict(text='Road 3', x=0.1, y=0.15, font_size=20, showarrow=False),
                    dict(text='Road 4', x=.63, y=0.15, font_size=20, showarrow=False)],
        showlegend=False
    )


    st.plotly_chart(fig)

    # Create another figure for Average Wait Time pie charts
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Road 1 - Average Waiting Time", "Road 2 - Average Waiting Time", "Road 3 - Average Waiting Time", "Road 4 - Average Waiting Time"],
        specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]  # Domain type for pie charts
    )

    # Pie charts for Average Wait Time by Lane for each Road
    for i in range(4):
        road_data = df1[(df1['Edge ID'] == df1['Edge ID'].unique()[i]) & (df1['Lane ID'] != 'All Lanes')]

        fig2.add_trace(go.Pie(
            labels=road_data['Lane ID'],
            values=road_data['Average Waiting Time (Lane) (s)'],
            marker=dict(colors=green_palette),  # Apply green color palette
            textinfo='label+value',
            hoverinfo='label+value+percent',
            textposition='inside',
            hole=0.4
        ), row=(i // 2) + 1, col=(i % 2) + 1)

    # Update layout to adjust the titles and remove legend
    # Update layout to add custom title for average wait time distribution
    fig2.update_layout(
        title_text="Average Waiting Time Distribution by Road",
        title_x=0.2,  # Adjust the title position
        title_font=dict(size=20, family='Arial', color='black', weight='normal'),
        annotations=[dict(text='Road 1', x=0.1, y=0.8, font_size=20, showarrow=False),
                    dict(text='Road 2', x=0.63, y=0.8, font_size=20, showarrow=False),
                    dict(text='Road 3', x=0.1, y=0.15, font_size=20, showarrow=False),
                    dict(text='Road 4', x=.63, y=0.15, font_size=20, showarrow=False)],
        showlegend=False
    )


    st.plotly_chart(fig2)

    # Bar chart for total vehicles and average wait time
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = ['Road 1', 'Road 2', 'Road 3', 'Road 4']

    # Extract the vehicle counts and average wait times for each road
    vehicle_counts = [df1[df1['Edge ID'] == road]['Total Vehicle Count'].iloc[-1] for road in df1['Edge ID'].unique()]
    average_wait_times = [df1[df1['Edge ID'] == road]['Average Waiting Time (Road) (s)'].dropna().values[0] for road in df1['Edge ID'].unique()]

    # Create clustered bar chart with green colors
    bar1 = ax3.bar(index, vehicle_counts, bar_width, label='Total Vehicles', color=green_palette[2], alpha=0.7)
    bar2 = ax3.bar(index, average_wait_times, bar_width, label='Average Wait Time (s)', color=green_palette[4], alpha=0.7, bottom=vehicle_counts)

    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.set_xticks(range(len(index)))
    ax3.set_xticklabels(index)

    # Add numbers inside the bars
    # For Total Vehicles (bar1) - Numbers at the top of the bar
    for i, bar in enumerate(bar1):
        yval = bar.get_height()  # Height of the vehicle count bar
        ax3.text(bar.get_x() + bar.get_width() / 2, yval - 5, round(yval, 2), ha='center', va='top')  # Adjust yval for positioning

    # For Average Wait Time (bar2) - Numbers above the stacked bars
    for i, bar in enumerate(bar2):
        yval = bar.get_height() + vehicle_counts[i] + 2  # Height of the stacked bar (vehicle + wait time)
        ax3.text(bar.get_x() + bar.get_width() / 2, yval, round(bar.get_height(), 2), ha='center', va='bottom')

    # Display the chart
    st.pyplot(fig3)


if page == "SUMO Simulation With Agent":
    st.markdown("""<div class="header"><h1> SUMO Simulation Dashboard</h1></div>""", unsafe_allow_html=True)
    sumo_video_path = "videos/After_Agent.mp4"
    st.video(sumo_video_path)

    df = pd.read_csv('data/simulation_data_last.csv')
    # Create a mapping for Edge ID to road names
    road_mapping = {edge: f"Road {i+1}" for i, edge in enumerate(df['Edge ID'].unique())}
    df['Road'] = df['Edge ID'].map(road_mapping)  # Create a new column 'Road'

    # Create a green color palette
    green_palette = ['#e0f2e9', '#b2e0d6', '#80c5b5', '#4fb99a', '#26a68a']  # Light to dark green

    # Subplots for four roads
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Road 1 - Vehicle Count", "Road 2 - Vehicle Count", "Road 3 - Vehicle Count", "Road 4 - Vehicle Count"],
        specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]  # Domain type for pie charts
    )

    for i in range(4):
        road_data = df[(df['Road'] == f"Road {i+1}") & (df['Lane ID'] != 'All Lanes')]

        fig.add_trace(go.Pie(
            labels=road_data['Lane ID'],
            values=road_data['Total Vehicle Count'],
            marker=dict(colors=['#b2e0d6', '#4fb99a', '#26a68a']),  # Green color palette
            textinfo='label+value',  # Show label and actual vehicle count instead of percentage
            hoverinfo='label+value',
            textposition='inside',
            hole=0.4
        ), row=(i // 2) + 1, col=(i % 2) + 1)

    # Update layout to adjust the titles and remove legend
    fig.update_layout(
        title_text="Vehicle Count Distribution by Road",
        title_x=0.25,  # Adjust the title position
        title_font=dict(size=20, family='Arial', color='black', weight='normal'),
        annotations=[dict(text='Road 1', x=0.1, y=0.8, font_size=20, showarrow=False),
                     dict(text='Road 2', x=0.63, y=0.8, font_size=20, showarrow=False),
                     dict(text='Road 3', x=0.1, y=0.15, font_size=20, showarrow=False),
                     dict(text='Road 4', x=.63, y=0.15, font_size=20, showarrow=False)],
        showlegend=False
    )

    st.plotly_chart(fig)

    # Create another figure for Average Wait Time pie charts
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Road 1 - Average Wait Time", "Road 2 - Average Wait Time", "Road 3 - Average Wait Time", "Road 4 - Average Wait Time"],
        specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]  # Domain type for pie charts
    )

    # Pie charts for Average Wait Time by Lane for each Road
    for i in range(4):
        road_data = df[(df['Road'] == f"Road {i+1}") & (df['Lane ID'] != 'All Lanes')]

        fig2.add_trace(go.Pie(
            labels=road_data['Lane ID'],
            values = road_data['avg_waiting_time'],
            marker=dict(colors=green_palette),  # Apply green color palette
            textinfo='label+value',
            hoverinfo='label+value+percent',
            textposition='inside',
            hole=0.4
        ), row=(i // 2) + 1, col=(i % 2) + 1)

    # Update layout to adjust the titles and remove legend
    fig2.update_layout(
        title_text="Average Wait Time Distribution by Road",
        title_x=0.2,  # Adjust the title position
        title_font=dict(size=20, family='Arial', color='black', weight='normal'),
        annotations=[dict(text='Road 1', x=0.1, y=0.8, font_size=20, showarrow=False),
                     dict(text='Road 2', x=0.63, y=0.8, font_size=20, showarrow=False),
                     dict(text='Road 3', x=0.1, y=0.15, font_size=20, showarrow=False),
                     dict(text='Road 4', x=.63, y=0.15, font_size=20, showarrow=False)],
        showlegend=False
    )

    st.plotly_chart(fig2)

    # Update bar chart to show 'Road' instead of 'Edge ID'
    avg_waiting_time_data = df[df['Lane ID'] == 'All Lanes']
    # st.bar_chart(avg_waiting_time_data[['Road', 'avg_waiting_time']].set_index('Road'))

    # Bar chart for total vehicles and average wait time for "All Lanes"
    # Bar chart for total vehicles and average wait time for "All Lanes"
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = [f"Road {i+1}" for i in range(4)]

    # Filter the data to include only rows where Lane ID is 'All Lanes'
    all_lanes_data = df[df['Lane ID'] == 'All Lanes']

    # Calculate the total vehicle count and average waiting time for each road
    vehicle_counts = [all_lanes_data[all_lanes_data['Road'] == road]['Total Vehicle Count'].sum() for road in index]
    average_wait_times = [all_lanes_data[all_lanes_data['Road'] == road]['avg_waiting_time'].mean() for road in index]

    # Create clustered bar chart
    bar1 = ax3.bar(index, vehicle_counts, bar_width, label='Total Vehicles', color=green_palette[2], alpha=0.7)
    bar2 = ax3.bar(index, average_wait_times, bar_width, label='Average Wait Time (s)', color=green_palette[4], alpha=0.7, bottom=vehicle_counts)

    # Set labels and legend
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.set_xticks(range(len(index)))
    ax3.set_xticklabels(index)

    # Add text at the bottom of each bar for total vehicle counts
    for i, bar in enumerate(bar1):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, 14, round(yval, 2), ha='center', va='top')  # Slightly below the bar

    # Add text on top of the stacked bars for average wait time
    for i, bar in enumerate(bar2):
        yval = bar.get_height() + vehicle_counts[i]
        ax3.text(bar.get_x() + bar.get_width() / 2, yval, round(bar.get_height(), 2), ha='center', va='bottom')

    # Display the bar chart
    st.pyplot(fig3)



    df1 = pd.read_csv('data/final_lane_road_data2.csv')

    # Check if 'Lane ID' exists in df1
    if 'Lane ID' in df1.columns:
        # Extract average waiting times for "All Lanes" from df1
        df1_all_lanes = df1[df1['Lane ID'] == 'All Lanes'][['Edge ID', 'Average Waiting Time (Road) (s)']]
        df1_all_lanes.rename(columns={'Average Waiting Time (Road) (s)': 'avg_waiting_time'}, inplace=True)
        df1_all_lanes['Simulation Type'] = 'Without Agent'
    else:
        st.error("'Lane ID' column not found in df1!")


    df2 = pd.read_csv('data/simulation_data_last.csv')
    # Check if 'Lane ID' exists in df2
    if 'Lane ID' in df.columns:
        # Extract average waiting times for "All Lanes" from df2
        df2_all_lanes = df[df['Lane ID'] == 'All Lanes'][['Edge ID', 'avg_waiting_time']]
        df2_all_lanes['Simulation Type'] = 'With Agent'
    else:
        st.error("'Lane ID' column not found in df2!")

# Combine average wait times from both pages
if 'df1_all_lanes' in locals() and 'df2_all_lanes' in locals():
    combined_data = pd.concat([df1_all_lanes, df2_all_lanes], ignore_index=True)

    # Convert avg_waiting_time to numeric, handling any possible errors
    combined_data['avg_waiting_time'] = pd.to_numeric(combined_data['avg_waiting_time'], errors='coerce')

    # Create a mapping for Edge ID to Road Name
    combined_data['Road'] = 'Road ' + (combined_data['Edge ID'].astype('category').cat.codes + 1).astype(str)


    # Create a bar plot to compare average wait times
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_plot = sns.barplot(data=combined_data, x='Road', y='avg_waiting_time', hue='Simulation Type', ax=ax, palette='pastel')

    # Set labels and title
    ax.set_title('Comparison of AWT With and Without Agent')
    ax.set_ylabel('Average Wait Time (s)')
    ax.set_xlabel('Road Number')

    # Add values inside the bars
    for p in bar_plot.patches:
        # Get the height of the bar (average waiting time)
        height = p.get_height()
        # Format the height to two decimal places
        formatted_height = f'{height:.2f}'  # Ensuring it's formatted correctly
        ax.annotate(formatted_height,
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=10, color='black',
                    xytext=(0, 5),  # Offset text from the top of the bar
                    textcoords='offset points')
    # Display the plot in Streamlit
    st.pyplot(fig)

