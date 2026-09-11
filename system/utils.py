import streamlit as st
import os
import numpy as np
import joblib
import cv2
from skimage.feature import hog

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

@st.cache_resource
def load_all_models():
    draw_dir  = os.path.join(PROJECT_ROOT, 'ai', 'drawing')
    gait_dir  = os.path.join(PROJECT_ROOT, 'ai', 'gait')
    voice_dir = os.path.join(PROJECT_ROOT, 'ai', 'voice')

    def _load(path):
        if os.path.exists(path):
            return joblib.load(path)
        return None

    return {
        'draw_model':   _load(os.path.join(draw_dir,  'drawing_model.pkl')),
        'draw_scaler':  _load(os.path.join(draw_dir,  'drawing_scaler.pkl')),
        'gait_model':   _load(os.path.join(gait_dir,  'gait_model.pkl')),
        'gait_scaler':  _load(os.path.join(gait_dir,  'gait_scaler.pkl')),
        'voice_model':  _load(os.path.join(voice_dir, 'voice_model.pkl')),
        'voice_scaler': _load(os.path.join(voice_dir, 'voice_scaler.pkl')),
        'voice_feats':  _load(os.path.join(voice_dir, 'voice_features.pkl')),
    }


def extract_hog(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    resized = cv2.resize(gray, (128, 128))
    features = hog(
        resized, orientations=9,
        pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm='L2-Hys', visualize=False, feature_vector=True,
    )
    return features


def extract_gait_features(df):
    total_left  = df.iloc[:, 16].astype(float)
    total_right = df.iloc[:, 17].astype(float)
    return np.array([
        total_left.mean(), total_right.mean(),
        total_left.std(),  total_right.std(),
        abs(total_left.mean() - total_right.mean()),
    ])


def extract_voice_features(wav_path):
    import parselmouth
    from parselmouth.praat import call
    import librosa

    sound = parselmouth.Sound(wav_path)
    pitch = call(sound, 'To Pitch', 0.0, 75, 600)
    pp    = call(sound, 'To PointProcess (periodic, cc)', 75, 600)

    j_pct  = call(pp, 'Get jitter (local)',           0, 0, 0.0001, 0.02, 1.3) * 100
    j_abs  = call(pp, 'Get jitter (local, absolute)', 0, 0, 0.0001, 0.02, 1.3)
    j_rap  = call(pp, 'Get jitter (rap)',              0, 0, 0.0001, 0.02, 1.3)
    j_ppq  = call(pp, 'Get jitter (ppq5)',             0, 0, 0.0001, 0.02, 1.3)
    j_ddp  = j_rap * 3

    s_loc   = call([sound, pp], 'Get shimmer (local)',    0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_db    = call([sound, pp], 'Get shimmer (local_dB)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_apq3  = call([sound, pp], 'Get shimmer (apq3)',     0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_apq5  = call([sound, pp], 'Get shimmer (apq5)',     0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_apq11 = call([sound, pp], 'Get shimmer (apq11)',    0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_dda   = s_apq3 * 3

    harmonicity = call(sound, 'To Harmonicity (cc)', 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, 'Get mean', 0, 0)
    nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0

    y_aud, sr = librosa.load(wav_path, sr=None)
    cent  = librosa.feature.spectral_centroid(y=y_aud, sr=sr)[0]
    band  = librosa.feature.spectral_bandwidth(y=y_aud, sr=sr)[0]
    dfa   = float(np.mean(band) / (np.mean(cent) + 1e-6))
    mfccs = librosa.feature.mfcc(y=y_aud, sr=sr, n_mfcc=20)
    rpde  = float(np.std(mfccs))

    pitch_vals = pitch.selected_array['frequency']
    pitch_vals = pitch_vals[pitch_vals > 0]
    if len(pitch_vals) > 0:
        hist, _ = np.histogram(pitch_vals, bins=20, density=True)
        hist = hist[hist > 0]
        ppe = float(-np.sum(hist * np.log(hist + 1e-6)))
    else:
        ppe = 0.5

    return {
        'Jitter(%)':    j_pct,  'Jitter(Abs)':    j_abs,
        'Jitter:RAP':   j_rap,  'Jitter:PPQ5':    j_ppq,  'Jitter:DDP':     j_ddp,
        'Shimmer':      s_loc,  'Shimmer(dB)':    s_db,
        'Shimmer:APQ3': s_apq3, 'Shimmer:APQ5':   s_apq5, 'Shimmer:APQ11': s_apq11,
        'Shimmer:DDA':  s_dda,  'NHR': nhr,       'HNR':  hnr,
        'RPDE': rpde,           'DFA': dfa,        'PPE':  ppe,
    }
