#!/usr/bin/env python
# coding: utf-8

# In[1]:
##################################################################################
"""All pakages here"""
#################################################################################
import sqlite3
import pandas as pd

from scipy.signal import welch
import numpy as np


from antropy import sample_entropy
from antropy import app_entropy

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression



from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score



from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from scipy.optimize import linear_sum_assignment
import numpy as np


from sklearn.model_selection import train_test_split, RandomizedSearchCV


#################################################################################
# Connect to the database present in sqlite3
conn = sqlite3.connect("Hall/a1c.db")

# List all tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

# Read data from the 'clinical' table
df = pd.read_sql("SELECT * FROM clinical LIMIT 10;", conn)  # This will help you find the A1C column name

# If A1C column is named exactly 'A1C', extract all rows
df_a1c = pd.read_sql("SELECT userID, A1C FROM clinical;", conn)
conn.close()
############################################################################
# Load the TSV file
df_OGTT_FBG = pd.read_csv('Hall/ogtt_2hours_FBG.tsv', sep='\t')
df_fbg = df_OGTT_FBG[df_OGTT_FBG['parameter'] == 'FBG']

# Show the first few rows
df_ogtt120= df_OGTT_FBG[(df_OGTT_FBG['parameter']=='OGTT') & (df_OGTT_FBG['timepoint_mins']==120)]

# Rename columns to avoid name clashes
df_fbg = df_fbg.rename(columns={'value': 'FBG'})
df_ogtt120 = df_ogtt120.rename(columns={'value': 'OGTT_120'})

# Merge step-by-step
df_merge = pd.merge(df_a1c, df_fbg[['userID', 'FBG']], on='userID', how='left')
df_merge = pd.merge(df_merge, df_ogtt120[['userID', 'OGTT_120']], on='userID', how='left')
###############################################################################
"""Funtions to classify the data into healthy, prediabetic and diabetic"""
################################################################################

def classify_patient(row):
    classifications = []

    # HbA1c
    if pd.notna(row['A1C']):
        if row['A1C'] >= 6.5:
            classifications.append('diabetic')
        elif row['A1C'] >= 5.7:
            classifications.append('prediabetic')
        else:
            classifications.append('healthy')

    # FBG
    if pd.notna(row['FBG']):
        if row['FBG'] >= 126:
            classifications.append('diabetic')
        elif row['FBG'] >= 100:
            classifications.append('prediabetic')
        else:
            classifications.append('healthy')

    # OGTT
    if pd.notna(row['OGTT_120']):
        if row['OGTT_120'] >= 200:
            classifications.append('diabetic')
        elif row['OGTT_120'] >= 140:
            classifications.append('prediabetic')
        else:
            classifications.append('healthy')

    # Use most severe classification
    if 'diabetic' in classifications:
        return 'diabetic'
    elif 'prediabetic' in classifications:
        return 'prediabetic'
    elif 'healthy' in classifications:
        return 'healthy'
    else:
        return 'unknown'

##########################################################################
df_merge['classification'] = df_merge.apply(classify_patient, axis=1)
#########################################################################
"""Load CGM data"""
##########################################################################
df_cgm = pd.read_csv('Hall/cgm.s010', delimiter='\t', engine='python')  # Try guessing the delimiter
df_cgm['Time'] = pd.to_datetime(df_cgm['DisplayTime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_cgm['TimeFormatted'] = df_cgm['Time'].dt.strftime("%d-%m-%Y %H:%M")
df_cgm['Day'] = df_cgm['Time'].dt.date

###########################################################################

# Separate CGM data for each subject and attach classification
subject_classes = df_merge.set_index('userID')['classification'].to_dict()

# Add classification to CGM dataframe
df_cgm['classification'] = df_cgm['subjectId'].map(subject_classes)
df_cgm['Glucose'] = pd.to_numeric(df_cgm['GlucoseValue'], errors='coerce')
# Split into groups
df_diabetic = df_cgm[df_cgm['classification'] == 'diabetic']
df_prediabetic = df_cgm[df_cgm['classification'] == 'prediabetic']
df_healthy = df_cgm[df_cgm['classification'] == 'healthy']

################################################################################
# In[10]:


# Group each class DataFrame by subjectId
# Group by subject and get only the first 5 subjects
healthy_subjects = {subject: group for i, (subject, group) in enumerate(df_healthy.groupby('subjectId')) if i < 6}
prediabetic_subjects = {subject: group for i, (subject, group) in enumerate(df_prediabetic.groupby('subjectId')) if i < 6}
diabetic_subjects = {subject: group for i, (subject, group) in enumerate(df_diabetic.groupby('subjectId')) if i < 6}



##########################################################################

for subject_id, df_subject in healthy_subjects.items():
    unique_times = df_subject['Day'].nunique()
########################################################################




#########################################################
"""Data Down sample from 5 minutes per sample to 15 minutes"""
############################################################
def data_downsample(data):

    imported_data = {}
    
    for sub in data.keys():
        df = data[sub].copy()
    
        df['Time'] = pd.to_datetime(df['InternalTime'], errors='coerce')
        df = df.sort_values('Time').reset_index(drop=True)
        # Step 3: Calculate time difference between consecutive rows
        df['TimeDiff'] = df['Time'].diff()
        
        # Step 4: Identify gaps greater than 15 minutes
        gap_threshold = pd.Timedelta(seconds=900)  # 900 seconds = 15 minutes
        df['Segment'] = (df['TimeDiff'] > gap_threshold).cumsum()
        
        # Step 5: Split into segments
        segments = [group.reset_index(drop=True) for _, group in df.groupby('Segment')]
            # Step 1: Define the range of desired time points, every 900 seconds
        data_per_subject=[]
        for seg in segments: 
            start_time = seg['Time'].iloc[0]
            end_time = seg['Time'].iloc[-1]
            time_grid = pd.date_range(start=start_time, end=end_time, freq='900s')  # 900 seconds = 15 min
            
            # Step 2: Use merge_asof to pick the nearest row for each 15-min time
            # It assumes df is sorted by Time
            df_downsampled = pd.merge_asof(
                pd.DataFrame({'TargetTime': time_grid}),
                seg,
                left_on='TargetTime',
                right_on='Time',
                direction='nearest'
            )
            
            # Optional: Drop any duplicates (if nearest rows were reused)
            df_downsampled = df_downsampled.drop_duplicates(subset='Time')
        
            # Optional: Reset index for clean output
            data_per_subject.append(df_downsampled.reset_index(drop=True))
    
        imported_data[sub] = data_per_subject
        
    return imported_data




# Group each class DataFrame by subjectId
healthy_subjects_down =  data_downsample(healthy_subjects)
prediabetic_subjects_down =  data_downsample(prediabetic_subjects)
diabetic_subjects_down =  data_downsample(diabetic_subjects)
###########################################################################################

###########################################################################
"""Extract Frequency domain Features"""
##########################################################################

def compute_frequency_features(glucose_series):
    fs = 1 / 900  # 15-minute sampling = 1 sample every 900 seconds
    glucose = glucose_series.dropna().values

    if len(glucose) < 4:
        return {
            'Total_Power': np.nan,
            'Dominant_Frequency': np.nan,
            'Spectral_Entropy': np.nan,
            'ULF_Power': np.nan,
            'VLF_Power': np.nan
        }

    # Compute Power Spectral Density
    freqs, psd = welch(glucose, fs=fs, nperseg=len(glucose))

    # Frequency bands based on new Nyquist (≈ 0.00056 Hz)
    ulf_power = np.sum(psd[(freqs >= 0.0000) & (freqs < 0.0003)])
    vlf_power = np.sum(psd[(freqs >= 0.0003) & (freqs <= 0.00056)])  # Nyquist limit

    total_power = np.sum(psd)
    dom_freq = freqs[np.argmax(psd)]

    # Normalize PSD for entropy calculation
    psd_norm = psd / (psd.sum() + 1e-12)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    return {
        'Total_Power': total_power,
        'Dominant_Frequency': dom_freq,
        'Spectral_Entropy': spectral_entropy,
        'ULF_Power': ulf_power,
        'VLF_Power': vlf_power
    }

def compute_signal_change_features(glucose_series, threshold_spike=20, threshold_change=5):
    glucose = glucose_series.dropna().values
    if len(glucose) < 2:
        return {
            'Spike_Count': np.nan,
            'Change_Frequency': np.nan,
            'Mean_RoC': np.nan,
            'Max_RoC': np.nan,
            'Std_RoC': np.nan
        }

    diff = np.diff(glucose)
    spike_count = np.sum(np.abs(diff) >= threshold_spike)
    change_freq = np.sum(np.abs(diff) >= threshold_change) / len(glucose)

    return {
        'Spike_Count': spike_count,
        'Change_Frequency': change_freq,
        'Mean_RoC': np.mean(diff),
        'Max_RoC': np.max(diff),
        'Std_RoC': np.std(diff)
    }

######################################################################
"""Time Domain Features"""
######################################################################

def interdaycv(df):
    return (np.std(df['Glucose']) / np.mean(df['Glucose'])) * 100

def interdaysd(df):
    return np.std(df['Glucose'])

def intradaycv(df):
    values = [interdaycv(df[df['Day'] == day]) for day in pd.unique(df['Day'])]
    return np.mean(values), np.median(values), np.std(values)

def intradaysd(df):
    values = [np.std(df[df['Day'] == day]['Glucose']) for day in pd.unique(df['Day']) if not df[df['Day'] == day]['Glucose'].empty]
    return np.mean(values), np.median(values), np.std(values)

def TIR(df, sd=1, sr=15):
    up = np.mean(df['Glucose']) + sd * np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd * np.std(df['Glucose'])
    return len(df[(df['Glucose'] <= up) & (df['Glucose'] >= dw)]) * sr

def TOR(df, sd=1, sr=15):
    up = np.mean(df['Glucose']) + sd * np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd * np.std(df['Glucose'])
    return len(df[(df['Glucose'] >= up) | (df['Glucose'] <= dw)]) * sr

def POR(df, sd=1, sr=15):
    """
    Percent time outside range
    """
    if df.empty or len(df) * sr == 0:
        return np.nan  # or 0 if you prefer

    tor = TOR(df, sd, sr)
    return (tor / (len(df) * sr)) * 100

def PIR(df, sd=1, sr=15):
    """
    Computes and returns the percent time inside glucose range.

    Args:
        df (pd.DataFrame): DataFrame containing 'Glucose' values.
        sd (int, optional): Standard deviation multiplier for range. Default is 1.
        sr (int, optional): Sampling rate in minutes. Default is 15.

    Returns:
        float: Percent time in range (%), or NaN if df is empty or invalid.
    """
    if df.empty or len(df) * sr == 0:
        return np.nan  # Avoid division by zero

    tir = TIR(df, sd, sr)
    return (tir / (len(df) * sr)) * 100

def MGE(df, sd=1):
    up = np.mean(df['Glucose']) + sd * np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd * np.std(df['Glucose'])
    return np.mean(df['Glucose'][(df['Glucose'] >= up) | (df['Glucose'] <= dw)])


def MAGE(df, std=1):
    """
        Computes and returns the mean amplitude of glucose excursions
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MAGE (float): the mean amplitude of glucose excursions 
        Refs:
            Sneh Gajiwala: https://github.com/snehG0205/NCSA_genomics/tree/2bfbb87c9c872b1458ef3597d9fb2e56ac13ad64
            
    """
        
    #extracting glucose values and incdices
    glucose = df['Glucose'].tolist()
    ix = [1*i for i in range(len(glucose))]
    stdev = std
    
    # local minima & maxima
    a = np.diff(np.sign(np.diff(glucose))).nonzero()[0] + 1      
    # local min
    valleys = (np.diff(np.sign(np.diff(glucose))) > 0).nonzero()[0] + 1 
    # local max
    peaks = (np.diff(np.sign(np.diff(glucose))) < 0).nonzero()[0] + 1         
    # +1 -- diff reduces original index number

    #store local minima and maxima -> identify + remove turning points
    excursion_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
    k=0
    for i in range(len(peaks)):
        excursion_points.loc[k] = [peaks[i]] + [df['Time'].iloc[k]] + [df['Glucose'].iloc[k]] + ["P"]
        k+=1

    for i in range(len(valleys)):
        excursion_points.loc[k] = [valleys[i]] + [df['Time'].iloc[k]] + [df['Glucose'].iloc[k]] + ["V"]
        k+=1

    excursion_points = excursion_points.sort_values(by=['Index'])
    excursion_points = excursion_points.reset_index(drop=True)


    # selecting turning points
    turning_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
    k=0
    for i in range(stdev,len(excursion_points.Index)-stdev):
        positions = [i-stdev,i,i+stdev]
        for j in range(0,len(positions)-1):
            if(excursion_points.Type[positions[j]] == excursion_points.Type[positions[j+1]]):
                if(excursion_points.Type[positions[j]]=='P'):
                    if excursion_points.Glucose[positions[j]]>=excursion_points.Glucose[positions[j+1]]:
                        turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k+=1
                    else:
                        turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k+=1
                else:
                    if excursion_points.Glucose[positions[j]]<=excursion_points.Glucose[positions[j+1]]:
                        turning_points.loc[k] = excursion_points.loc[positions[j]]
                        k+=1
                    else:
                        turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k+=1

    if len(turning_points.index)<10:
        turning_points = excursion_points.copy()
        excursion_count = len(excursion_points.index)
    else:
        excursion_count = len(excursion_points.index)/2


    turning_points = turning_points.drop_duplicates(subset= "Index", keep= "first")
    turning_points=turning_points.reset_index(drop=True)
    excursion_points = excursion_points[excursion_points.Index.isin(turning_points.Index) == False]
    excursion_points = excursion_points.reset_index(drop=True)

    # calculating MAGE
    if excursion_count == 0:
        return np.nan  # or 0 or raise a warning
    else:
        mage = turning_points.Glucose.sum() / excursion_count
        return round(mage, 3)
   




def MGN(df, sd=1):
    up = np.mean(df['Glucose']) + sd * np.std(df['Glucose'])
    dw = np.mean(df['Glucose']) - sd * np.std(df['Glucose'])
    return np.mean(df['Glucose'][(df['Glucose'] <= up) & (df['Glucose'] >= dw)])

def J_index(df):
    return 0.001 * ((np.mean(df['Glucose']) + np.std(df['Glucose'])) ** 2)

def LBGI_HBGI(df):
    f = (np.log(df['Glucose']) ** 1.084) - 5.381
    rl = [22.77 * (i ** 2) if i <= 0 else 0 for i in f]
    rh = [22.77 * (i ** 2) if i > 0 else 0 for i in f]
    return np.mean(rl), np.mean(rh), rh, rl

def LBGI(df):
    return LBGI_HBGI(df)[0]

def HBGI(df):
    return LBGI_HBGI(df)[1]

def ADRR(df):
    """
        Computes and returns the average daily risk range, an assessment of total daily glucose variations within risk space
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            ADRRx (float): average daily risk range
            
    """
    ADRRl = []
    for i in pd.unique(df['Day']):
        LBGI, HBGI, rh, rl = LBGI_HBGI(df[df['Day']==i])
        LR = np.max(rl)
        HR = np.max(rh)
        ADRRl.append(LR+HR)

    ADRRx = np.mean(ADRRl)
    return ADRRx

def uniquevalfilter(df, value):
    diff = abs(df[df['Minfrommid'] == value]['Glucose'].diff())
    return np.nanmean(diff)

def MODD(df):
    df['Minfrommid'] = df['Time'].dt.hour * 60 + df['Time'].dt.minute
    modds = [uniquevalfilter(df, val) for val in df['Minfrommid'].unique()]
    return np.nanmean([val for val in modds if val != 0])

def CONGA24(df):
    df['Minfrommid'] = df['Time'].dt.hour * 60 + df['Time'].dt.minute
    conga = [uniquevalfilter(df, val) for val in df['Minfrommid'].unique()]
    return np.nanstd([val for val in conga if val != 0])

def GMI(df):
    return 3.31 + 0.02392 * np.mean(df['Glucose'])

def eA1c(df):
    return (46.7 + np.mean(df['Glucose'])) / 28.7

def summary(df):
    return (
        np.nanmean(df['Glucose']),
        np.nanmedian(df['Glucose']),
        np.nanmin(df['Glucose']),
        np.nanmax(df['Glucose']),
        np.nanpercentile(df['Glucose'], 25),
        np.nanpercentile(df['Glucose'], 75),
    )


def shannon_entropy(x):
    x = np.array(x)
    value, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def compute_mobility(x, delta_t=15):
    x = np.array(x)
    dx = np.diff(x) / delta_t  # Δxi / Δt
    mobility = np.sqrt(np.var(dx) / np.var(x))
    return mobility
#####################################################################################


# In[18]:

##################################################################################
"""Extract all features inside the dictionary"""
def feature_extraction(data):
    all_data_one_subject = []

    for i in range(len(data)):
        df = data[i].dropna()

        if df.empty or 'Glucose' not in df.columns or df['Glucose'].dropna().empty:
            continue  # skip empty or invalid data

        # Build dictionary of scalar metrics
        metrics_dict = {
            'TIR': TIR(df),
            'TOR': TOR(df),
            'interdaycv': interdaycv(df),
            'interdaysd': interdaysd(df),
         
            'POR': POR(df),
            'PIR': PIR(df),
            'MGE': MGE(df),
            'MAGE': MAGE(df),
            'MGN': MGN(df),
          
            'LBGI': LBGI(df),
            'HBGI': HBGI(df),
            'ADRR': ADRR(df),
            'MODD': MODD(df),
            'CONGA24': CONGA24(df),
            'GMI': GMI(df),
            'eA1c': eA1c(df),
        }
        # Add intra-day values (tuple unpacking)
        intra_cv_mean, intra_cv_median, intra_cv_sd = intradaycv(df)
        metrics_dict.update({
            'Intraday_CV_mean': intra_cv_mean,
            'Intraday_CV_median': intra_cv_median,
            'Intraday_CV_sd': intra_cv_sd
        })
        
        meanG, medianG, minG, maxG, Q1G, Q3G  = summary(df)
        metrics_dict.update({
            'meanG': meanG,
            'medianG': medianG,
            'minG': minG,
            'maxG': maxG,
            'Q1G': Q1G,
            'Q3G': Q3G
        })
    
        
        intra_sd_mean, intra_sd_median, intra_sd_sd = intradaysd(df)
        metrics_dict.update({
            'Intraday_SD_mean': intra_sd_mean,
            'Intraday_SD_median': intra_sd_median,
            'Intraday_SD_sd': intra_sd_sd
        })
        # Add robust summary statistics
        glucose = df['Glucose'].dropna()
        if glucose.empty:
            metrics_dict.update({
                'mean': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'percentile': np.nan
            })
        else:
            metrics_dict.update({
                'mean': np.nanmean(glucose),
                'median': np.nanmedian(glucose),
                'min': np.nanmin(glucose),
                'max': np.nanmax(glucose),
                'percentile': np.nanpercentile(glucose, 25)
            })

        # Add time-series and signal-based features
        metrics_dict.update({
            'mobility': compute_mobility(glucose),
            'shannon_entropy': shannon_entropy(glucose),
            'sampleentropy': sample_entropy(glucose)
        })

        freq_feats = compute_frequency_features(glucose)
        metrics_dict.update(freq_feats)

        signal_change_feats = compute_signal_change_features(glucose)
        metrics_dict.update(signal_change_feats)

        # Convert to one-row DataFrame and append
        metrics_df = pd.DataFrame([metrics_dict])
        all_data_one_subject.append(metrics_df)

    return pd.concat(all_data_one_subject, ignore_index=True)
#####################################################################################



features_duke_diabetic=[]
for sub in diabetic_subjects_down.keys():
    features_duke_diabetic.append(feature_extraction(diabetic_subjects_down[sub]))

diabetic= pd.concat(features_duke_diabetic)
diabetic['class']='diabetic'
features_duke_prediabetic=[]
for sub in prediabetic_subjects_down.keys():
    features_duke_prediabetic.append(feature_extraction(prediabetic_subjects_down[sub]))
prediabetic= pd.concat(features_duke_prediabetic)
prediabetic['class']='prediabetic'
features_duke_healthy=[]
for sub in healthy_subjects_down.keys():
    features_duke_healthy.append(feature_extraction(healthy_subjects_down[sub]))
healthy= pd.concat(features_duke_healthy)
healthy['class']='healthy'

##########################################################################
all_data = pd.concat([diabetic, prediabetic, healthy_take])

##########################################################################


label_mapping = {
    "healthy": 1,
    "prediabetic": 2,
    "diabetic": 3
}

all_data["class"] = all_data["class"].map(label_mapping)

########################################################################
# Step 1: Find rows that will be dropped
rows_with_na = all_data[all_data.isnull().any(axis=1)]

# Step 2: See exactly which columns have NaNs in those rows
nan_details = rows_with_na.isnull()

# Step 3: Print or analyze which rows and columns had NaNs
for idx, row in nan_details.iterrows():
    cols_with_nan = row[row].index.tolist()
    print(f"Row {idx} dropped due to NaNs in columns: {cols_with_nan}")




all_data = all_data.drop(['MODD', 'CONGA24'], axis=1)

############################################################################
# In[38]:


all_data = all_data.replace([np.inf, -np.inf], np.nan)

all_data= all_data.dropna()


##############################################################################



# Prepare data
X = all_data.drop('class', axis=1)
y = all_data['class']
feature_names = list(X.columns)  # Convert to list to safely index

# Balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Select top 20 features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y_resampled)

# Get mask of selected features
mask = selector.get_support()

# Use list comprehension for safe feature selection
selected_features = [feature_names[i] for i, keep in enumerate(mask) if keep]
print("Selected Features:")
print(selected_features)

# Print feature scores, sorted
feature_scores = selector.scores_
feature_score_dict = {name: score for name, score in zip(feature_names, feature_scores)}
sorted_scores = sorted(feature_score_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Scores (sorted):")
for name, score in sorted_scores:
    print(f"{name}: {score:.2f}")


###################################################################




# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


# Prepare data
X = all_data.drop('class', axis=1)
y = all_data['class']


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Use Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Track F1 scores
f1_scores = []
num_features_list = list(range(1, X.shape[1] + 1, 2))  # Step by 2 to reduce computation

for n_features in num_features_list:
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    y_pred = rfe.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(score)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features_list, f1_scores, marker='o')
plt.xlabel("Number of Selected Features")
plt.ylabel("F1 Score (weighted)")
plt.title("F1 Score vs Number of Features (RFE with Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.show()

#######################################################################


# In[43]:






# In[ ]:
###########################################################################


# Prepare data
X = all_data.drop('class', axis=1)
y = all_data['class']

# Encode target labels if necessary
if y.dtype == object or y.dtype.name == 'category':
    y = pd.factorize(y)[0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Model for RFE
model = LogisticRegression(max_iter=1000)

# Track F1 scores
f1_scores = []
num_features_list = list(range(1, X.shape[1] + 1))

for n_features in num_features_list:
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    y_pred = rfe.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(score)

# Plot F1 score vs number of features
plt.figure(figsize=(10, 6))
plt.plot(num_features_list, f1_scores, marker='o')
plt.xlabel("Number of Selected Features")
plt.ylabel("F1 Score (weighted)")
plt.title("F1 Score vs Number of Features (RFE)")
plt.grid(True)
plt.tight_layout()
plt.show()


#######################################################


X_top3=all_data[['Dominant_Frequency',
'shannon_entropy',
'mobility',
'Change_Frequency',
'maxG',
'max',
]]
y_true=all_data['class']


# 3. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top3)

# 4. KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)
conf_mat = confusion_matrix(y_true, y_pred)

# 2. Align cluster labels to true labels
row_ind, col_ind = linear_sum_assignment(-conf_mat)
label_map = {pred: true for pred, true in zip(col_ind, row_ind)}
y_pred_aligned = np.array([label_map[label] for label in y_pred])

# 3. Final confusion matrix
conf_mat_aligned = confusion_matrix(y_true, y_pred_aligned)
print("Confusion Matrix (aligned):\n", conf_mat_aligned)

# 4. Accuracy
acc = accuracy_score(y_true, y_pred_aligned)
print("Accuracy:", acc)

# 5. F1, Precision, Recall (aka Sensitivity)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_aligned, digits=4))

# 6. Sensitivity and Specificity (manual per class)
for i in range(conf_mat_aligned.shape[0]):
    TP = conf_mat_aligned[i, i]
    FN = conf_mat_aligned[i, :].sum() - TP
    FP = conf_mat_aligned[:, i].sum() - TP
    TN = conf_mat_aligned.sum() - (TP + FP + FN)

    sensitivity = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0

    print(f"\nClass {i}:")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")



# ##########################################################

# Step 1: Prepare data
X = all_data.drop('class', axis=1)
y = all_data['class']
feature_names = list(X.columns)

# Step 2: Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 3: Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Step 4: Feature selection
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y_resampled)

# Track selected features
mask = selector.get_support()
selected_features = [feature_names[i] for i, keep in enumerate(mask) if keep]
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y_resampled, stratify=y_resampled, test_size=0.1, random_state=42)

# Step 6: Define parameter space for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,  # Try 20 different combinations
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Step 7: Fit and evaluate
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Predict and report
y_pred = best_model.predict(X_test)
print("Best Parameters:", search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:
######################################################


# Step 1: Prepare data
X = all_data.drop('class', axis=1)
y = all_data['class']
feature_names = list(X.columns)

# Step 2: Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 3: Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Step 4: Feature selection
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y_resampled)

# Track selected features
mask = selector.get_support()
selected_features = [feature_names[i] for i, keep in enumerate(mask) if keep]
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y_resampled, stratify=y_resampled, random_state=42)

# Step 6: Define parameter space for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,  # Try 20 different combinations
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Step 7: Fit and evaluate
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Predict and report
y_pred = best_model.predict(X_test)
print("Best Parameters:", search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




