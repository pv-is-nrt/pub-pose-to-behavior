"""
Publication utilities for SLEAP behavior classification analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sleap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_sleap_data(pose_slp_path):
    """Load SLEAP file and extract pose data with track information."""
    labels = sleap.load_file(str(pose_slp_path))
    print(f"Loaded {len(labels)} labeled frames in {len(labels.videos)} videos")
    
    pose_data = []
    for labeled_frame in labels:
        frame_idx = labeled_frame.frame_idx
        video_id = int(Path(str(labeled_frame.video.filename)).stem.replace('video', ''))
        
        for instance in labeled_frame.instances:
            if instance.track is None:
                continue
                
            track_id = instance.track.name
            
            # Extract landmark coordinates
            landmarks = {}
            for node_name, point in zip(labels.skeleton.node_names, instance.points):
                landmarks[f"{node_name}.x"] = point.x if not np.isnan(point.x) else np.nan
                landmarks[f"{node_name}.y"] = point.y if not np.isnan(point.y) else np.nan
                landmarks[f"{node_name}.score"] = getattr(point, 'score', np.nan)
            
            record = {
                'video_id': video_id,
                'track': track_id,
                'frame_idx': frame_idx + 1,  # Convert to 1-based indexing
                'instance_type': type(instance).__name__,
                'instance.score': getattr(instance, 'score', 1.0),
                **landmarks
            }
            pose_data.append(record)
    
    df = pd.DataFrame(pose_data)
    df = df.sort_values(by=['video_id', 'frame_idx', 'track'], ignore_index=True)
    return df

def clean_pose_data(df):
    """Clean pose data by removing missing coordinates and problematic tracks."""
    print(f"Initial dataset: {df.shape}")
    
    # Remove rows with missing coordinates
    coordinate_cols = ['head.x', 'head.y', 'neck.x', 'neck.y', 'center.x', 'center.y', 'tail.x', 'tail.y']
    missing_before = df[coordinate_cols].isnull().any(axis=1).sum()
    print(f"Rows with missing coordinates: {missing_before}")
    
    df = df.dropna(subset=coordinate_cols)
    print(f"After removing missing coordinates: {df.shape}")
    
    # Remove video 1 right side animal (poor visibility)
    df = df[~((df['video_id'] == 1) & (df['track'] == 'track_0'))]
    print(f"After removing video 1 right side animal: {df.shape}")
    
    return df

def add_behavior_labels(df, behavior_labels):
    """Add behavior labels to dataframe based on frame ranges."""
    
    # Track mapping based on spatial analysis
    track_mapping = {
        1: {'L': 'track_1', 'R': 'track_0'},
        2: {'L': 'track_0', 'R': 'track_1'},
        3: {'L': 'track_0', 'R': 'track_1'},
        5: {'L': 'track_1', 'R': 'track_0'}
    }
    
    def get_behavior_for_frame(video_id, track, frame_idx):
        # Determine if this is L or R track
        if video_id in track_mapping:
            if track == track_mapping[video_id]['L']:
                position = 'L'
            elif track == track_mapping[video_id]['R']:
                position = 'R'
            else:
                return None
        else:
            position = 'L'  # Single track videos
        
        behavior_key = f"video{video_id}_{position}"
        if behavior_key not in behavior_labels:
            return None
        
        # Check which behavior this frame belongs to
        for behavior, ranges in behavior_labels[behavior_key].items():
            for start, end in ranges:
                if start <= frame_idx <= end:
                    return behavior
        
        return 'other'  # Frames not in any specific behavior
    
    print("Adding behavior labels...")
    df['behavior'] = df.apply(
        lambda row: get_behavior_for_frame(row['video_id'], row['track'], row['frame_idx']), 
        axis=1
    )
    
    # Remove unlabeled and 'other' frames for 3-class classification
    df = df[df['behavior'].notna()].copy()
    df = df[df['behavior'] != 'other'].copy()
    
    print(f"Final labeled dataset: {df.shape}")
    print(f"Behavior distribution:\n{df['behavior'].value_counts()}")
    
    return df

def engineer_core_features(df):
    """Engineer 12 core features for publication analysis."""
    df_features = df.copy()
    
    print("Engineering core features...")
    
    # 1. Center-normalized coordinates (6 features)
    df_features['head_rel_x'] = df_features['head.x'] - df_features['center.x']
    df_features['head_rel_y'] = df_features['head.y'] - df_features['center.y']
    df_features['neck_rel_x'] = df_features['neck.x'] - df_features['center.x']
    df_features['neck_rel_y'] = df_features['neck.y'] - df_features['center.y']
    df_features['tail_rel_x'] = df_features['tail.x'] - df_features['center.x']
    df_features['tail_rel_y'] = df_features['tail.y'] - df_features['center.y']
    
    # 2. Inter-landmark distances (4 features)
    df_features['dist_head_neck'] = np.sqrt((df_features['head.x'] - df_features['neck.x'])**2 + 
                                          (df_features['head.y'] - df_features['neck.y'])**2)
    df_features['dist_head_center'] = np.sqrt((df_features['head.x'] - df_features['center.x'])**2 + 
                                            (df_features['head.y'] - df_features['center.y'])**2)
    df_features['dist_neck_center'] = np.sqrt((df_features['neck.x'] - df_features['center.x'])**2 + 
                                            (df_features['neck.y'] - df_features['center.y'])**2)
    df_features['dist_center_tail'] = np.sqrt((df_features['center.x'] - df_features['tail.x'])**2 + 
                                            (df_features['center.y'] - df_features['tail.y'])**2)
    
    # 3. Body curvature (1 feature)
    v1_x = df_features['head.x'] - df_features['center.x']
    v1_y = df_features['head.y'] - df_features['center.y']
    v2_x = df_features['tail.x'] - df_features['center.x']
    v2_y = df_features['tail.y'] - df_features['center.y']
    
    dot_product = v1_x * v2_x + v1_y * v2_y
    mag1 = np.sqrt(v1_x**2 + v1_y**2)
    mag2 = np.sqrt(v2_x**2 + v2_y**2)
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1, 1)
    df_features['body_curvature'] = np.abs(np.arccos(cos_angle))
    
    # 4. Activity level (1 feature) - velocity-based
    df_features = df_features.sort_values(['video_id', 'track', 'frame_idx']).reset_index(drop=True)
    
    # Center of mass
    df_features['com_x'] = (df_features['head.x'] + df_features['neck.x'] + df_features['center.x'] + df_features['tail.x']) / 4
    df_features['com_y'] = (df_features['head.y'] + df_features['neck.y'] + df_features['center.y'] + df_features['tail.y']) / 4
    
    # Velocity calculation for consecutive frames
    prev_com_x = df_features.groupby(['video_id', 'track'])['com_x'].shift(1)
    prev_com_y = df_features.groupby(['video_id', 'track'])['com_y'].shift(1)
    prev_frame = df_features.groupby(['video_id', 'track'])['frame_idx'].shift(1)
    
    consecutive_mask = (df_features['frame_idx'] == prev_frame + 1)
    velocity_x = np.where(consecutive_mask, df_features['com_x'] - prev_com_x, 0)
    velocity_y = np.where(consecutive_mask, df_features['com_y'] - prev_com_y, 0)
    velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
    
    # Activity level as 10-frame rolling average
    df_features['velocity_magnitude'] = velocity_magnitude
    df_features['activity_level'] = df_features.groupby(['video_id', 'track'])['velocity_magnitude'].rolling(
        window=10, center=True, min_periods=1
    ).mean().reset_index(drop=True)
    
    # Clean up temporary columns
    df_features.drop(['com_x', 'com_y', 'velocity_magnitude'], axis=1, inplace=True)
    
    # Define core feature set (12 features)
    core_features = [
        'head_rel_x', 'head_rel_y', 'neck_rel_x', 'neck_rel_y', 'tail_rel_x', 'tail_rel_y',
        'dist_head_neck', 'dist_head_center', 'dist_neck_center', 'dist_center_tail',
        'body_curvature', 'activity_level'
    ]
    
    print(f"Created {len(core_features)} core features for publication")
    return df_features, core_features

def scale_features(df, feature_cols):
    """Apply StandardScaler to specified features."""
    df_scaled = df.copy()
    
    # Features to scale (exclude body_curvature which is already normalized)
    features_to_scale = [f for f in feature_cols if f != 'body_curvature']
    
    scaler = StandardScaler()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    
    print(f"Scaled {len(features_to_scale)} features using StandardScaler")
    return df_scaled, scaler

def get_dataset_summary(df):
    """Generate comprehensive dataset summary."""
    summary = {
        'total_frames': len(df),
        'total_videos': df['video_id'].nunique(),
        'total_animals': df.groupby(['video_id', 'track']).ngroups,
        'behavior_counts': df['behavior'].value_counts().to_dict(),
        'video_frame_counts': df.groupby('video_id').size().to_dict(),
        'animal_frame_counts': df.groupby(['video_id', 'track']).size().to_dict()
    }
    return summary

def print_dataset_summary(summary):
    """Print formatted dataset summary."""
    print("DATASET SUMMARY")
    print("=" * 40)
    print(f"Total frames: {summary['total_frames']:,}")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Total animals: {summary['total_animals']}")
    
    print(f"\nBehavior distribution:")
    for behavior, count in summary['behavior_counts'].items():
        pct = count / summary['total_frames'] * 100
        print(f"  {behavior}: {count:,} frames ({pct:.1f}%)")
    
    print(f"\nFrames per video:")
    for video_id, count in summary['video_frame_counts'].items():
        print(f"  Video {video_id}: {count:,} frames")

def evaluate_models(df_scaled, core_features, test_size=0.2, random_state=42):
    """Evaluate multiple ML models for behavior classification."""
    
    # Prepare data
    X = df_scaled[core_features]
    y = df_scaled['behavior']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Define models with parameters appropriate for dataset size
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=random_state),
        'SVM': SVC(kernel='rbf', random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1000, random_state=random_state)
    }
    
    # Train and evaluate models
    results = {}
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': model
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
    
    return results, X_test, y_test


def create_temporal_blocks(df, block_size=15, test_size=0.2, random_state=42):
    """
    Create temporal block-based train/test split for behavioral time series.
    
    This function divides behavioral sequences into consecutive temporal blocks
    and performs stratified sampling at the block level to ensure:
    1. Temporal independence between train/test sets
    2. Balanced representation of each behavior class
    3. Preservation of temporal structure within blocks
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with columns: video_id, track, frame_idx, behavior, and features
    block_size : int, default=15
        Number of consecutive frames per temporal block
    test_size : float, default=0.2
        Proportion of blocks to use for testing (0.0 to 1.0)
    random_state : int, default=42
        Random seed for reproducible splits
        
    Returns:
    --------
    train_indices : list
        Row indices for training data
    test_indices : list
        Row indices for testing data
    block_info : dict
        Information about the blocking process and distributions
    """
    
    np.random.seed(random_state)
    
    # Sort data by video, track, and frame for proper temporal ordering
    df_sorted = df.sort_values(['video_id', 'track', 'frame_idx']).reset_index(drop=True)
    
    # Create blocks for each animal individually
    train_indices = []
    test_indices = []
    block_stats = {'total_blocks': 0, 'train_blocks': 0, 'test_blocks': 0}
    behavior_block_counts = {'train': {}, 'test': {}}
    
    print("TEMPORAL BLOCK SPLIT CREATION")
    print("=" * 50)
    print(f"Block size: {block_size} frames")
    print(f"Test proportion: {test_size:.1%}")
    
    # Process each animal separately
    for (video_id, track), animal_data in df_sorted.groupby(['video_id', 'track']):
        animal_id = f"video{video_id}_{track}"
        print(f"\nProcessing {animal_id} ({len(animal_data)} frames)...")
        
        # Create blocks for this animal
        animal_blocks = []
        for i in range(0, len(animal_data), block_size):
            block_data = animal_data.iloc[i:i+block_size]
            
            # Only keep blocks that have uniform behavior (no mixed behaviors)
            if len(block_data['behavior'].unique()) == 1:
                behavior = block_data['behavior'].iloc[0]
                animal_blocks.append({
                    'indices': block_data.index.tolist(),
                    'behavior': behavior,
                    'size': len(block_data),
                    'animal_id': animal_id
                })
        
        # Group blocks by behavior for this animal
        behavior_blocks = {}
        for block in animal_blocks:
            behavior = block['behavior']
            if behavior not in behavior_blocks:
                behavior_blocks[behavior] = []
            behavior_blocks[behavior].append(block)
        
        # Stratified sampling of blocks for each behavior
        for behavior, blocks in behavior_blocks.items():
            n_test_blocks = max(1, int(len(blocks) * test_size))
            n_train_blocks = len(blocks) - n_test_blocks
            
            # Randomly select test blocks
            test_block_indices = np.random.choice(len(blocks), n_test_blocks, replace=False)
            
            for i, block in enumerate(blocks):
                if i in test_block_indices:
                    test_indices.extend(block['indices'])
                    if behavior not in behavior_block_counts['test']:
                        behavior_block_counts['test'][behavior] = 0
                    behavior_block_counts['test'][behavior] += 1
                else:
                    train_indices.extend(block['indices'])
                    if behavior not in behavior_block_counts['train']:
                        behavior_block_counts['train'][behavior] = 0
                    behavior_block_counts['train'][behavior] += 1
            
            print(f"  {behavior}: {len(blocks)} blocks → {n_train_blocks} train, {n_test_blocks} test")
            block_stats['total_blocks'] += len(blocks)
            block_stats['train_blocks'] += n_train_blocks
            block_stats['test_blocks'] += n_test_blocks
    
    # Create final train/test datasets
    train_data = df_sorted.loc[train_indices]
    test_data = df_sorted.loc[test_indices]
    
    # Compile block information
    block_info = {
        'block_size': block_size,
        'test_size': test_size,
        'total_blocks': block_stats['total_blocks'],
        'train_blocks': block_stats['train_blocks'],
        'test_blocks': block_stats['test_blocks'],
        'train_frames': len(train_data),
        'test_frames': len(test_data),
        'train_behavior_counts': train_data['behavior'].value_counts().to_dict(),
        'test_behavior_counts': test_data['behavior'].value_counts().to_dict(),
        'block_behavior_counts': behavior_block_counts
    }
    
    print(f"\nTEMPORAL BLOCK SPLIT SUMMARY")
    print("-" * 30)
    print(f"Total blocks created: {block_info['total_blocks']}")
    print(f"Training blocks: {block_info['train_blocks']} ({block_info['train_frames']} frames)")
    print(f"Test blocks: {block_info['test_blocks']} ({block_info['test_frames']} frames)")
    print(f"Train behavior distribution: {block_info['train_behavior_counts']}")
    print(f"Test behavior distribution: {block_info['test_behavior_counts']}")
    
    return train_indices, test_indices, block_info


def evaluate_models_temporal_blocks(df, features, block_size=15, test_size=0.2, random_state=42):
    """
    Evaluate ML models using temporal block-based train/test split.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Scaled dataset with features and behavior labels
    features : list
        List of feature column names to use for training
    block_size : int, default=15
        Number of consecutive frames per temporal block
    test_size : float, default=0.2
        Proportion of blocks for testing
    random_state : int, default=42
        Random seed for reproducible results
        
    Returns:
    --------
    results : dict
        Model performance metrics
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test labels
    block_info : dict
        Information about temporal blocking
    """
    
    # Create temporal block split
    train_indices, test_indices, block_info = create_temporal_blocks(
        df, block_size=block_size, test_size=test_size, random_state=random_state
    )
    
    # Prepare features and labels
    X_train = df.loc[train_indices, features]
    X_test = df.loc[test_indices, features]
    y_train = df.loc[train_indices, 'behavior']
    y_test = df.loc[test_indices, 'behavior']
    
    # Define models with parameters appropriate for dataset size
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=random_state),
        'SVM': SVC(kernel='rbf', random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1000, random_state=random_state)
    }
    
    # Train and evaluate models
    results = {}
    print("\nMODEL EVALUATION RESULTS (Temporal Blocks)")
    print("=" * 50)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': model
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
    
    return results, X_test, y_test, block_info


def evaluate_models_animal_split(df, features, test_animals, random_state=42):
    """
    Evaluate multiple ML models using animal-level split for cross-individual generalization.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe with scaled features
    features : list
        List of feature column names to use
    test_animals : list
        List of animal IDs to use as test set
    random_state : int
        Random state for model initialization
        
    Returns
    -------
    results : dict
        Dictionary with model performance metrics
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    """
    
    # Create animal_id column if not exists
    if 'animal_id' not in df.columns:
        df = df.copy()
        df['animal_id'] = df['video_id'].astype(str) + '_track_' + df['track'].astype(str)
    
    # Create train/test split based on animals
    test_mask = df['animal_id'].isin(test_animals)
    
    X_train = df[~test_mask][features]
    X_test = df[test_mask][features] 
    y_train = df[~test_mask]['behavior']
    y_test = df[test_mask]['behavior']
    
    # Define models with parameters appropriate for dataset size
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=random_state),
        'SVM': SVC(kernel='rbf', random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1000, random_state=random_state)
    }
    
    # Train and evaluate models
    results = {}
    print("\nMODEL EVALUATION RESULTS (Animal-Level Split)")
    print("=" * 50)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': model
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
    
    return results, X_test, y_test


def create_behavior_timeline_visualization(df_scaled, core_features, test_animals, behavior_colors, 
                                         figure_width=6.5, save_dpi=300, output_dir=None):
    """
    Create comprehensive timeline visualization showing ground truth and predictions from all validation strategies.
    
    Parameters
    ----------
    df_scaled : pd.DataFrame
        Scaled dataset with features and behaviors
    core_features : list
        List of all feature column names
    test_animals : list
        List of test animal IDs for animal-level split
    behavior_colors : dict
        Color scheme dictionary with train/test colors for each behavior
    figure_width : float
        Figure width in inches
    save_dpi : int
        DPI for save
    output_dir : Path or str
        Directory to save figures
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    predictions_dict : dict
        Dictionary containing all model predictions
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    print("CREATING BEHAVIOR TIMELINE VISUALIZATION")
    print("=" * 50)
    
    # Ensure animal_id column exists
    if 'animal_id' not in df_scaled.columns:
        df_scaled = df_scaled.copy()
        df_scaled['animal_id'] = df_scaled['video_id'].astype(str) + '_track_' + df_scaled['track'].astype(str)
    
    # Get unique animals sorted by video and track (proper numerical sorting)
    def sort_key(animal_id):
        # Extract video number and track from animal_id format like "1_track_track_1"
        parts = animal_id.split('_track_')
        video_num = int(parts[0])
        track_part = parts[1] if len(parts) > 1 else 'track_0'
        track_num = int(track_part.split('_')[-1]) if track_part.split('_')[-1].isdigit() else 0
        return (video_num, track_num)
    
    animals = sorted(df_scaled['animal_id'].unique(), key=sort_key)
    # reverse ordering
    animals = animals[::-1]
    n_animals = len(animals)

    print("animals:", animals)
    
    print(f"Generating predictions for {n_animals} animals using best performing models...")
    
    # Initialize predictions storage
    predictions_dict = {}
    
    # 1. STRATIFIED RANDOM PREDICTIONS (Random Forest)
    print("\n1. Training Stratified Random Forest...")
    X = df_scaled[core_features]
    y = df_scaled['behavior']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    stratified_pred = rf_model.predict(X)
    predictions_dict['strat. random'] = stratified_pred
    
    # Create stratified test mask
    test_indices = X_test.index
    stratified_test_mask = df_scaled.index.isin(test_indices)
    
    # 2. TEMPORAL BLOCK PREDICTIONS (Random Forest)
    print("2. Training Temporal Block Random Forest...")
    
    # Create temporal blocks
    df_blocks = []
    for animal_id in animals:
        animal_data = df_scaled[df_scaled['animal_id'] == animal_id].copy()
        animal_data = animal_data.sort_values('frame_idx')
        
        # Create 15-frame blocks
        block_size = 15
        n_blocks = len(animal_data) // block_size
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_data = animal_data.iloc[start_idx:end_idx]
            
            # Only keep uniform behavior blocks
            behaviors = block_data['behavior'].unique()
            if len(behaviors) == 1 and pd.notna(behaviors[0]):
                block_info = {
                    'animal_id': animal_id,
                    'block_id': f"{animal_id}_block_{i}",
                    'behavior': behaviors[0],
                    'start_frame': block_data['frame_idx'].min(),
                    'end_frame': block_data['frame_idx'].max(),
                    'indices': block_data.index.tolist()
                }
                df_blocks.append(block_info)
    
    # Split blocks by behavior
    blocks_df = pd.DataFrame(df_blocks)
    temporal_test_indices = []
    
    for behavior in ['sleeping', 'feeding', 'crawling']:
        behavior_blocks = blocks_df[blocks_df['behavior'] == behavior]
        if len(behavior_blocks) > 0:
            n_test_blocks = max(1, int(0.2 * len(behavior_blocks)))
            test_blocks = behavior_blocks.sample(n=n_test_blocks, random_state=42)
            for _, block in test_blocks.iterrows():
                temporal_test_indices.extend(block['indices'])
    
    temporal_train_indices = [idx for idx in df_scaled.index if idx not in temporal_test_indices]
    
    # Train temporal model
    X_temporal_train = df_scaled.loc[temporal_train_indices, core_features]
    y_temporal_train = df_scaled.loc[temporal_train_indices, 'behavior']
    
    rf_temporal = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_temporal.fit(X_temporal_train, y_temporal_train)
    temporal_pred = rf_temporal.predict(X)
    predictions_dict['temporal block'] = temporal_pred
    
    # Create temporal test mask
    temporal_test_mask = df_scaled.index.isin(temporal_test_indices)
    
    # 3. ANIMAL-LEVEL PREDICTIONS (Neural Network)
    print("3. Training Animal-Level Neural Network...")
    
    # Create animal-level split
    test_mask = df_scaled['animal_id'].isin(test_animals)
    X_animal_train = df_scaled[~test_mask][core_features]
    y_animal_train = df_scaled[~test_mask]['behavior']
    
    nn_model = MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1000, random_state=42)
    nn_model.fit(X_animal_train, y_animal_train)
    animal_pred = nn_model.predict(X)
    predictions_dict['animal-level'] = animal_pred
    
    # Create animal-level test mask
    animal_test_mask = test_mask
    
    print("4. Creating timeline visualization...")
    
    # Define row positions with gaps
    bar_height = 0.09  # Reduced bar height
    small_gap = 0.02   # Small gap between bars of same animal
    large_gap = 0.15    # Gap between animals
    rows_per_animal = 4  # Ground truth + 3 predictions
    
    # Calculate total height needed
    total_height = n_animals * (rows_per_animal * bar_height + (rows_per_animal - 1) * small_gap) + (n_animals - 1) * large_gap

    print("total height:", total_height)
    
    # Set up the plot with publication style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(figure_width, total_height))
    
    current_y = 0  # Start from bottom (y=0)
    for animal_idx, animal_id in enumerate(animals):
        animal_data = df_scaled[df_scaled['animal_id'] == animal_id].copy()
        animal_data = animal_data.sort_values('frame_idx')
        
        if len(animal_data) == 0:
            continue
            
        frames = animal_data['frame_idx'].values
        behaviors = animal_data['behavior'].values
        
        # Calculate base y position for this animal
        if animal_idx > 0:
            current_y += large_gap  # Gap between animals
        
        # Row labels and data
        rows_data = [
            ('animal-level', predictions_dict['animal-level'][animal_data.index], animal_test_mask[animal_data.index]),
            ('temporal block', predictions_dict['temporal block'][animal_data.index], temporal_test_mask[animal_data.index]),
            ('strat. random', predictions_dict['strat. random'][animal_data.index], stratified_test_mask[animal_data.index]),
            ('ground truth', behaviors, None)
        ]
        
        # Plot each row for this animal
        for row_idx, (row_label, row_behaviors, test_mask_row) in enumerate(rows_data):
            y_pos = current_y + row_idx * (bar_height + small_gap)
            
            # Plot behavior segments
            for i, (frame, behavior) in enumerate(zip(frames, row_behaviors)):
                # Determine if this frame is in test set
                if test_mask_row is not None:
                    if hasattr(test_mask_row, 'iloc'):
                        is_test_frame = test_mask_row.iloc[i]
                    else:
                        is_test_frame = test_mask_row[i]
                else:
                    is_test_frame = False
                
                # Get color (train vs test)
                if row_label == 'ground truth':
                    color = behavior_colors[behavior]['train']
                else:
                    color = behavior_colors[behavior]['test'] if is_test_frame else behavior_colors[behavior]['train']
                
                # Plot frame
                ax.barh(y_pos, 1, left=frame, height=bar_height, color=color)
        
        # Add animal label with incremented animal number
        video_num = animal_id.split('_track_')[0]
        track_part = animal_id.split('_track_')[1] if '_track_' in animal_id else 'track_0'
        animal_num = int(track_part.split('_')[-1]) if track_part.split('_')[-1].isdigit() else 0
        animal_num += 1  # Increment by 1
        
        animal_label = f"video {video_num}, animal {animal_num}"
        # position the label above the top bar for the animal
        label_y = current_y + (rows_per_animal * bar_height + (rows_per_animal - 1) * small_gap)
        # axis label
        ax.text(10, label_y, animal_label, ha='left', va='center', fontsize=6, fontfamily='Arial', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0'))

        # Update current_y for next animal
        current_y += rows_per_animal * bar_height + (rows_per_animal - 1) * small_gap
    
    # Add row labels for each animal
    current_y = 0
    for animal_idx in range(n_animals):
        if animal_idx > 0:
            current_y += large_gap
        
        row_labels = ['animal-level', 'temporal block', 'strat. random', 'ground truth']
        for row_idx, label in enumerate(row_labels):
            y_pos = current_y + row_idx * (bar_height + small_gap)
            ax.text(-10, y_pos, label, ha='right', va='center', fontsize=5, fontfamily='Arial')
        
        current_y += rows_per_animal * bar_height + (rows_per_animal - 1) * small_gap
    
    # Create legend
    legend_elements = []
    for behavior, colors in behavior_colors.items():
        legend_elements.append(patches.Patch(color=colors['train'], label=f'{behavior.capitalize()} (train)'))
        legend_elements.append(patches.Patch(color=colors['test'], label=f'{behavior.capitalize()} (test)'))
    
    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.95, 0.55), prop={'family': 'Arial', 'size': 7})
    
    # Set labels and formatting
    ax.set_xlabel('Frame Index', fontsize=7, fontfamily='Arial')
    # ax.set_ylabel('Animals and Predictions', fontsize=7, fontfamily='Arial')
    # ax.set_title('Behavior Classification Timeline: Ground Truth vs Model Predictions', fontsize=8, fontfamily='Arial', pad=5)
    
    # Set axis limits and formatting
    ax.set_xlim(0, df_scaled['frame_idx'].max() + 10)
    ax.set_ylim(-0.1, total_height + 0.15)
    
    # Add gridlines - major and minor
    ax.grid(True, alpha=0.7, axis='x', which='major', linewidth=0.5)
    ax.grid(True, alpha=0.7, axis='x', which='minor', linestyle=':', linewidth=0.5)
    ax.set_xticks(range(0, int(df_scaled['frame_idx'].max()) + 1, 200))  # Major ticks every 200 frames
    ax.set_xticks(range(0, int(df_scaled['frame_idx'].max()) + 1, 50), minor=True)  # Minor ticks every 50 frames
    
    ax.set_yticks([])
    
    # Format x-axis with Arial font
    ax.tick_params(axis='x', labelsize=6)
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    
    plt.tight_layout()
    
    # Save figures if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        
        # Save
        fig.savefig(output_dir / 'behavior_timeline.png', 
                   dpi=save_dpi, bbox_inches='tight', facecolor='white')
        print(f"Timeline saved: {output_dir / 'behavior_timeline.png'}")

    print(f"\nTimeline visualization completed!")
    print(f"- {n_animals} animals visualized")
    print(f"- 4 rows per animal (ground truth + 3 predictions)")
    print(f"- Test frames shown in light colors, train frames in dark colors")
    
    return fig, predictions_dict


import pickle
from sklearn.metrics import confusion_matrix


def save_best_model_confusion_matrix(models_dict, X_test, y_test, split_type, output_dir, behavior_order=['sleeping', 'feeding', 'crawling']):
    """
    Identify the best performing model and save its confusion matrix.
    
    Parameters:
    models_dict : dict - Dictionary of model results from evaluation
    X_test : array - Test features  
    y_test : array - Test labels
    split_type : str - Type of validation split ('stratified_random', 'temporal_blocks', 'animal_level')
    output_dir : Path - Directory to save results
    behavior_order : list - Order of behaviors for confusion matrix
    
    Returns:
    best_model_name : str - Name of the best performing model
    cm : array - Confusion matrix of the best model
    accuracy : float - Accuracy of the best model
    """
    # Find best model by accuracy
    best_model_name = max(models_dict.keys(), key=lambda k: models_dict[k]['accuracy'])
    best_model = models_dict[best_model_name]['model']
    best_accuracy = models_dict[best_model_name]['accuracy']
    
    print(f"Best model for {split_type}: {best_model_name} (Accuracy: {best_accuracy:.3f})")
    
    # Generate predictions with the best model
    y_pred = best_model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=behavior_order)
    
    # Save the confusion matrix and metadata
    cm_data = {
        'split_type': split_type,
        'best_model': best_model_name,
        'accuracy': best_accuracy,
        'confusion_matrix': cm,
        'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred,
        'test_size': len(y_test),
        'behavior_order': behavior_order
    }
    
    # Save to pickle file
    cm_file = output_dir / f'{split_type}_best_confusion_matrix.pkl'
    with open(cm_file, 'wb') as f:
        pickle.dump(cm_data, f)
    
    print(f"Confusion matrix saved: {cm_file}")
    return best_model_name, cm, best_accuracy


def plot_confusion_matrices_from_saved(output_dir, behavior_order=['sleeping', 'feeding', 'crawling']):
    """
    Load saved confusion matrices and create publication-quality plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set font to Arial for consistency
    plt.rcParams['font.family'] = 'Arial'
    
    # Load confusion matrix data
    split_types = ['stratified_random', 'temporal_blocks', 'animal_level'] 
    split_names = ['stratified random', 'temporal block', 'animal-level']
    
    # Map behaviors to single letters
    behavior_labels = ['S', 'F', 'C']  # S=sleeping, F=feeding, C=crawling
    
    # Create figure - vertical layout (3 rows, 1 column)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
    
    # Use the purple color similar to visualization
    cmap_color = "#580B4F"
    # Create a custom colormap based on the purple
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['white', cmap_color]
    custom_cmap = LinearSegmentedColormap.from_list('custom_purple', colors)
    
    for idx, (split_type, split_name) in enumerate(zip(split_types, split_names)):
        ax = axes[idx]
        
        try:
            # Load confusion matrix data
            cm_file = output_dir / f'{split_type}_best_confusion_matrix.pkl'
            with open(cm_file, 'rb') as f:
                cm_data = pickle.load(f)
            
            cm = cm_data['confusion_matrix']
            best_model = cm_data['best_model']
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotations with counts and percentages
            annotations = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    percent = cm_percent[i, j]
                    annotations[i, j] = f'{count}\n{percent:.1f}%'
                    # annotations[i, j] = f'{count}'
                    # annotations[i, j] = f'{percent:.1f}%'
                    
            
            # Create heatmap
            sns.heatmap(cm, annot=annotations, fmt='', cmap=custom_cmap, 
                       xticklabels=behavior_labels, yticklabels=behavior_labels,
                       ax=ax, cbar=False, annot_kws={'fontsize': 8, 'fontfamily': 'Arial', 'fontweight': 'bold'})
            
            # Customize subplot
            ax.set_xlabel('predicted', fontsize=8, fontfamily='Arial')
            ax.set_ylabel('ground truth', fontsize=8, fontfamily='Arial')
            
            # Set tick labels
            ax.set_xticklabels(behavior_labels, fontsize=8, fontfamily='Arial', fontweight='bold')
            ax.set_yticklabels(behavior_labels, fontsize=8, fontfamily='Arial', fontweight='bold')

            # show outlines for the cm
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

            # make the spines thicker
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # Add model name as title of the subplot
            ax.set_title(f'{split_name}', fontsize=8, fontfamily='Arial', pad=10)

        except FileNotFoundError:
            ax.text(0.5, 0.5, f'{split_name}\nNot computed', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=7, fontfamily='Arial')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'confusion_matrices_best_models.png'
    plt.savefig(plot_path, dpi=1200, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrices plot saved: {plot_path}")
    
    plt.show()