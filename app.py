import streamlit as st
import os
import librosa
import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
import pandas as pd

TARGET_SR = 16000          
NEIGHBORHOOD_SIZE = 7      
DYNAMIC_OFFSET_DB = 15     
database_folder = 'EE200 Project Song Database'

@st.cache_resource
def build_cloud_database():
    if not os.path.exists(database_folder):
        return {}, {}
    
    audio_files = sorted([f for f in os.listdir(database_folder) if f.endswith('.mp3')])
    song_database = {}
    song_mapping = {idx: os.path.splitext(f)[0] for idx, f in enumerate(audio_files)}
    
    for song_idx, filename in enumerate(audio_files):
        song_path = os.path.join(database_folder, filename)
        y, fs = librosa.load(song_path, sr=TARGET_SR)
        _, _, Sxx = signal.spectrogram(y, fs, nperseg=2048)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        local_max = maximum_filter(Sxx_db, size=NEIGHBORHOOD_SIZE) == Sxx_db
        peaks_mask = local_max & (Sxx_db > (np.mean(Sxx_db) + DYNAMIC_OFFSET_DB))
        f_coor, t_coor = np.where(peaks_mask)
        peaks = list(zip(t_coor, f_coor))
        
        num_peaks = len(peaks)
        for i in range(num_peaks):
            t1, f1 = peaks[i]
            for j in range(i + 1, min(i + 15, num_peaks)): 
                t2, f2 = peaks[j]
                dt = t2 - t1
                if 1 <= dt <= 30: 
                    hash_key = (f1, f2, dt)
                    if hash_key not in song_database:
                        song_database[hash_key] = []
                    song_database[hash_key].append((song_idx, t1))
    return song_database, song_mapping

song_database, song_mapping = build_cloud_database()

def pipeline_and_match(query_audio_path):
    y_query, fs = librosa.load(query_audio_path, sr=TARGET_SR)
    frequencies, times, Sxx = signal.spectrogram(y_query, fs, nperseg=2048)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    local_max = maximum_filter(Sxx_db, size=NEIGHBORHOOD_SIZE) == Sxx_db
    peaks_mask = local_max & (Sxx_db > (np.mean(Sxx_db) + DYNAMIC_OFFSET_DB))
    freq_indices, time_indices = np.where(peaks_mask)
    query_peaks = list(zip(time_indices, freq_indices))
    
    song_offsets = {song_idx: [] for song_idx in song_mapping.keys()}
    num_peaks = len(query_peaks)
    
    for i in range(num_peaks):
        t1_q, f1 = query_peaks[i]
        for j in range(i + 1, min(i + 15, num_peaks)):
            t2_q, f2 = query_peaks[j]
            dt = t2_q - t1_q
            if 1 <= dt <= 30:
                hash_key = (f1, f2, dt)
                if hash_key in song_database:
                    for song_idx, t1_s in song_database[hash_key]:
                        song_offsets[song_idx].append(t1_s - t1_q)
                        
    best_song_idx, max_peak_votes, best_offsets_list = -1, 0, []
    for song_idx, offsets in song_offsets.items():
        if len(offsets) == 0: continue
        counts, _ = np.histogram(offsets, bins=np.arange(min(offsets)-1, max(offsets)+2, 1))
        current_max_peak = np.max(counts)
        if current_max_peak > max_peak_votes:
            max_peak_votes = current_max_peak
            best_song_idx = song_idx
            best_offsets_list = offsets

    predicted_song = song_mapping[best_song_idx] if best_song_idx != -1 else "Unknown Track"
    return predicted_song, max_peak_votes, best_offsets_list, times, frequencies, Sxx_db, time_indices, freq_indices

st.set_page_config(page_title="EE200 Matcher Engine", page_icon="🎵", layout="wide")
st.title("EE200 Audio Identification Panel")

if not song_database:
    st.error("Audio library directory missing from current directory workspace branch.")
else:
    mode = st.sidebar.radio("Select Application Mode:", ["(i) Single-Clip Mode", "(ii) Batch Mode"])

    if mode == "(i) Single-Clip Mode":
        st.subheader("Single-Clip Diagnostic Identification Viewer")
        uploaded_file = st.file_uploader("Upload query clip:", type=["mp3", "wav"])
        
        if uploaded_file is not None:
            temp_path = "temp_user_query.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            pred, votes, offsets, times, frequencies, Sxx_db, t_idx, f_idx = pipeline_and_match(temp_path)
            
            st.metric(label="Predicted Song Match Result", value=pred)
            st.metric(label="Matching Confidence Score", value=f"{votes} Alignment Votes")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='magma')
            axes[0].set_title("1. Dense 2D Spectrogram Input")
            axes[0].set_ylim(0, 4000)
            axes[0].set_ylabel("Frequency (Hz)")
            axes[0].set_xlabel("Time (s)")

            axes[1].scatter(times[t_idx], frequencies[f_idx], color='cyan', s=10, marker='o')
            axes[1].set_title(f"2. Extracted Constellation Map ({len(t_idx)} Peaks)")
            axes[1].set_ylim(0, 4000)
            axes[1].set_xlabel("Time (s)")
            axes[1].set_facecolor('black')

            if len(offsets) > 0:
                axes[2].hist(offsets, bins=np.arange(min(offsets)-1, max(offsets)+2, 1), color='royalblue', edgecolor='black')
                axes[2].set_title("3. Time-Offset Alignment Histogram")
                axes[2].set_xlabel("Time Offset Bin")
                axes[2].set_ylabel("Vote Count")
            st.pyplot(fig)
            os.remove(temp_path)

    elif mode == "(ii) Batch Mode":
        st.subheader("Automated Batch Identification Processor")
        uploaded_files = st.file_uploader("Upload query clips sequentially:", type=["mp3", "wav"], accept_multiple_files=True)
        
        if uploaded_files:
            results = []
            for i, up_file in enumerate(uploaded_files):
                temp_b_path = f"temp_batch_{i}.wav"
                with open(temp_b_path, "wb") as f:
                    f.write(up_file.getbuffer())
                pred, _, _, _, _, _, _, _ = pipeline_and_match(temp_b_path)
                
                results.append({"filename": up_file.name, "prediction": pred})
                os.remove(temp_b_path)
            
            df_results = pd.DataFrame(results)[["filename", "prediction"]]
            st.dataframe(df_results)
            st.download_button(
                label="Download official results.csv File",
                data=df_results.to_csv(index=False).encode('utf-8'),
                file_name="results.csv",
                mime="text/csv"
            )
