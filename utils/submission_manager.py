"""
Submission manager for rollercoaster submissions.
Handles saving and loading submissions using local JSON files.

Deployment Notes:
- Local development: Files are saved to 'submissions/' directory
- Streamlit Cloud: Files written at runtime are ephemeral (lost on restart)
- To persist submissions in deployment:
  1. Commit the 'submissions/' folder to git (for existing submissions)
  2. New submissions in deployment won't persist unless you use cloud storage
  3. For production, consider using a simple cloud storage solution
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict


def _get_submissions_dir():
    """Get the submissions directory path, creating it if needed."""
    # Try multiple possible locations
    possible_dirs = [
        'submissions',
        os.path.join(os.getcwd(), 'submissions'),
        os.path.join(os.path.dirname(__file__), '..', 'submissions'),
    ]
    
    for dir_path in possible_dirs:
        abs_path = os.path.abspath(dir_path)
        if os.path.exists(abs_path):
            return abs_path
    
    # Create the first directory if none exist
    submissions_dir = os.path.abspath(possible_dirs[0])
    os.makedirs(submissions_dir, exist_ok=True)
    return submissions_dir


def _is_deployment_environment():
    """Check if running in a deployment environment (e.g., Streamlit Cloud)."""
    # Streamlit Cloud sets this environment variable
    return os.getenv('STREAMLIT_SERVER_ENV') is not None or os.getenv('STREAMLIT_SHARING') is not None


def save_submission(
    submitter_name: str,
    geometry: Dict,
    score: float,
    safety_score: float,
    timestamp: Optional[str] = None
) -> bool:
    """
    Save a rollercoaster submission to local JSON file.
    
    Note: In deployment environments (Streamlit Cloud), files are ephemeral
    and will be lost on restart. For persistence, commit submissions/ to git
    or use cloud storage.
    
    Args:
        submitter_name: Name of the person submitting
        geometry: Dict with 'x', 'y', 'z' arrays (track coordinates)
        score: Fun rating score (0-5)
        safety_score: Safety score (0-5)
        timestamp: Optional timestamp string (defaults to current time)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        submissions_dir = _get_submissions_dir()
        
        # Generate unique ID and timestamp
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create submission ID from timestamp
        submission_id = timestamp.replace(':', '-').replace('.', '-')
        
        # Prepare submission data
        submission_data = {
            'submission_id': submission_id,
            'submitter_name': submitter_name,
            'timestamp': timestamp,
            'score': float(score),
            'safety_score': float(safety_score),
            'geometry': {
                'x': [float(v) for v in geometry['x']],
                'y': [float(v) for v in geometry['y']],
                'z': [float(v) for v in geometry.get('z', [0.0] * len(geometry['x']))]
            }
        }
        
        # Save individual submission file
        submission_file = os.path.join(submissions_dir, f'{submission_id}.json')
        with open(submission_file, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, indent=2)
        
        # Update the leaderboard index
        _update_leaderboard_index(submissions_dir, submission_data)
        
        # Warn in deployment if files won't persist
        if _is_deployment_environment():
            print("⚠️ Warning: Running in deployment. Submissions will be lost on restart. "
                  "Consider committing submissions/ to git or using cloud storage.")
        
        return True
        
    except Exception as e:
        print(f"Error saving submission: {e}")
        return False


def _update_leaderboard_index(submissions_dir: str, submission_data: Dict):
    """Update the leaderboard index file."""
    try:
        leaderboard_file = os.path.join(submissions_dir, 'leaderboard.json')
        existing_submissions = []
        
        # Load existing leaderboard if it exists
        if os.path.exists(leaderboard_file):
            try:
                with open(leaderboard_file, 'r', encoding='utf-8') as f:
                    leaderboard_data = json.load(f)
                    existing_submissions = leaderboard_data.get('submissions', [])
            except:
                # File exists but is corrupted, start fresh
                pass
        
        # Add new submission (only metadata, not full geometry)
        submission_metadata = {
            'submission_id': submission_data['submission_id'],
            'submitter_name': submission_data['submitter_name'],
            'timestamp': submission_data['timestamp'],
            'score': submission_data['score'],
            'safety_score': submission_data['safety_score']
        }
        
        existing_submissions.append(submission_metadata)
        
        # Sort by combined score (score + safety_score) descending, then by individual scores as tiebreaker
        existing_submissions.sort(key=lambda x: (x['score'] + x['safety_score'], x['score'], x['safety_score']), reverse=True)
        
        # Save updated leaderboard
        leaderboard_data = {
            'last_updated': datetime.now().isoformat(),
            'submissions': existing_submissions
        }
        
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=2)
        
    except Exception as e:
        print(f"Error updating leaderboard index: {e}")


def load_submissions() -> List[Dict]:
    """
    Load all submissions from local JSON files.
    
    Returns:
        List of submission dictionaries (metadata only, sorted by combined score: score + safety_score)
    """
    try:
        submissions_dir = _get_submissions_dir()
        leaderboard_file = os.path.join(submissions_dir, 'leaderboard.json')
        
        # Load leaderboard index
        if os.path.exists(leaderboard_file):
            try:
                with open(leaderboard_file, 'r', encoding='utf-8') as f:
                    leaderboard_data = json.load(f)
                    return leaderboard_data.get('submissions', [])
            except:
                return []
        
        return []
        
    except Exception as e:
        print(f"Error loading submissions: {e}")
        return []


def load_submission_geometry(submission_id: str) -> Optional[Dict]:
    """
    Load full geometry for a specific submission.
    
    Args:
        submission_id: ID of the submission
        
    Returns:
        Dict with geometry data, or None if not found
    """
    try:
        submissions_dir = _get_submissions_dir()
        submission_file = os.path.join(submissions_dir, f'{submission_id}.json')
        
        if os.path.exists(submission_file):
            with open(submission_file, 'r', encoding='utf-8') as f:
                submission_data = json.load(f)
                return submission_data.get('geometry')
        
        return None
        
    except Exception as e:
        print(f"Error loading submission geometry: {e}")
        return None


def add_submission_to_leaderboard(
    submitter_name: str,
    score: float,
    safety_score: float,
    submission_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Add a submission to the leaderboard without saving geometry.
    Useful for RFDB data or other sources without track geometry.
    
    Args:
        submitter_name: Name of the person/ride submitting
        score: Fun rating score (0-5)
        safety_score: Safety score (0-5)
        submission_id: Optional custom submission ID (auto-generated if None)
        timestamp: Optional timestamp string (defaults to current time)
        metadata: Optional additional metadata to store
        
    Returns:
        True if successful, False otherwise
    """
    try:
        submissions_dir = _get_submissions_dir()
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        if submission_id is None:
            submission_id = timestamp.replace(':', '-').replace('.', '-')
        
        # Create submission metadata
        submission_metadata = {
            'submission_id': submission_id,
            'submitter_name': submitter_name,
            'timestamp': timestamp,
            'score': float(score),
            'safety_score': float(safety_score)
        }
        
        # Add any additional metadata
        if metadata:
            submission_metadata.update(metadata)
        
        # Update leaderboard index
        leaderboard_file = os.path.join(submissions_dir, 'leaderboard.json')
        existing_submissions = []
        
        if os.path.exists(leaderboard_file):
            try:
                with open(leaderboard_file, 'r', encoding='utf-8') as f:
                    leaderboard_data = json.load(f)
                    existing_submissions = leaderboard_data.get('submissions', [])
            except:
                pass
        
        # Check if submission already exists
        existing_ids = {s.get('submission_id') for s in existing_submissions}
        if submission_id in existing_ids:
            return False  # Already exists
        
        # Add new submission
        existing_submissions.append(submission_metadata)
        
        # Sort by combined score (score + safety_score) descending, then by individual scores as tiebreaker
        existing_submissions.sort(key=lambda x: (x['score'] + x['safety_score'], x['score'], x['safety_score']), reverse=True)
        
        # Save updated leaderboard
        leaderboard_data = {
            'last_updated': datetime.now().isoformat(),
            'submissions': existing_submissions
        }
        
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error adding submission to leaderboard: {e}")
        return False


def update_submission_in_leaderboard(
    submission_id: str,
    submitter_name: Optional[str] = None,
    score: Optional[float] = None,
    safety_score: Optional[float] = None,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Update an existing submission in the leaderboard.
    
    Args:
        submission_id: ID of the submission to update
        submitter_name: Optional new submitter name
        score: Optional new fun rating score
        safety_score: Optional new safety score
        metadata: Optional additional metadata to update
        
    Returns:
        True if successful, False if submission not found
    """
    try:
        submissions_dir = _get_submissions_dir()
        leaderboard_file = os.path.join(submissions_dir, 'leaderboard.json')
        
        if not os.path.exists(leaderboard_file):
            return False
        
        # Load existing leaderboard
        with open(leaderboard_file, 'r', encoding='utf-8') as f:
            leaderboard_data = json.load(f)
            existing_submissions = leaderboard_data.get('submissions', [])
        
        # Find and update the submission
        found = False
        for sub in existing_submissions:
            if sub.get('submission_id') == submission_id:
                found = True
                if submitter_name is not None:
                    sub['submitter_name'] = submitter_name
                if score is not None:
                    sub['score'] = float(score)
                if safety_score is not None:
                    sub['safety_score'] = float(safety_score)
                if metadata:
                    sub.update(metadata)
                break
        
        if not found:
            return False
        
        # Sort by combined score (score + safety_score) descending, then by individual scores as tiebreaker
        existing_submissions.sort(key=lambda x: (x['score'] + x['safety_score'], x['score'], x['safety_score']), reverse=True)
        
        # Save updated leaderboard
        leaderboard_data = {
            'last_updated': datetime.now().isoformat(),
            'submissions': existing_submissions
        }
        
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error updating submission in leaderboard: {e}")
        return False


# Backward compatibility aliases (in case code still uses old function names)
save_submission_to_s3 = save_submission
load_submissions_from_s3 = load_submissions
load_submission_geometry_from_s3 = load_submission_geometry
