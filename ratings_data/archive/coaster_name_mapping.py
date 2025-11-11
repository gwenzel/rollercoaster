"""
Complete mapping dictionary between roller coaster names in ratings_data 
and acceleration data filenames.
"""

# Mapping from ratings_data coaster names to accel_data filenames
COASTER_NAME_MAPPING = {
    # Direct matches
    "El Toro": "ElToro",
    "Hyperia": "Hyperia",
    "Iron Gwazi": "IronGwazi",
    "Maverick": "Maverick",
    "Pantheon": "Pantheon",
    "Shambhala": "Shambhala",
    "Skyrush": "Skyrush",
    "Steel Vengeance": "SteelVeng",
    "Taiga": "Taiga",
    "Taron": "Taron",
    "Untamed": "Untamed",
    "VelociCoaster": "VelociCoas",  # Truncated in filename
    "Zadra": "Zadra",
    
    # Matches found through search
    "AlpenFury": "AlpenFury",  # Custom/unknown coaster
    "Anubis: The Ride": "Anubis",
    "ArieForce One": "ArieForce",
    "Lightning Rod": "Lightning",  # Most likely candidate
    "Pantherian": "Pantherian",  # Appears in both datasets
    "Ride to Happiness": "RidetoHa",  # Truncated
    "Twisted Colossus": "TwistedCo",  # Most likely - popular RMC
    "Wicked Cyclone": "WickedCyc",  # Truncated
}

# Reverse mapping (accel_data filename -> ratings_data name)
ACCEL_TO_RATINGS_NAME = {v: k for k, v in COASTER_NAME_MAPPING.items()}

# Alternative names for ambiguous cases
ALTERNATIVE_MAPPINGS = {
    "Lightning": ["Lightning Rod", "Lightning Run", "Lightning Racer (Thunder)", "Lightning Racer (Lightning)"],
    "TwistedCo": ["Twisted Colossus", "Twisted Cyclone", "Twisted Timbers"],
}

def get_accel_filename(coaster_name):
    """
    Get the acceleration data filename for a given coaster name from ratings data.
    
    Args:
        coaster_name: Name of the coaster from ratings_data
        
    Returns:
        str: Filename (without .csv) or None if not found
    """
    return COASTER_NAME_MAPPING.get(coaster_name)

def get_ratings_name(accel_filename):
    """
    Get the ratings data coaster name for a given acceleration filename.
    
    Args:
        accel_filename: Filename from accel_data (without .csv)
        
    Returns:
        str: Coaster name from ratings_data or None if not found
    """
    return ACCEL_TO_RATINGS_NAME.get(accel_filename)

def print_mapping_summary():
    """Print a summary of the mapping."""
    print("="*70)
    print("ROLLER COASTER NAME MAPPING")
    print("="*70)
    print(f"\nTotal mapped coasters: {len(COASTER_NAME_MAPPING)}")
    print("\nRatings Data Name → Accel Data Filename:")
    print("-"*70)
    for ratings_name, accel_name in sorted(COASTER_NAME_MAPPING.items()):
        print(f"  {ratings_name:30s} → {accel_name}.csv")
    
    if ALTERNATIVE_MAPPINGS:
        print("\n" + "="*70)
        print("AMBIGUOUS MAPPINGS (multiple possible matches):")
        print("="*70)
        for accel_name, alternatives in ALTERNATIVE_MAPPINGS.items():
            print(f"\n{accel_name}.csv could be:")
            for alt in alternatives:
                print(f"  - {alt}")

if __name__ == "__main__":
    print_mapping_summary()
    
    # Example usage
    print("\n" + "="*70)
    print("EXAMPLE USAGE:")
    print("="*70)
    print(f'\nget_accel_filename("Iron Gwazi") → {get_accel_filename("Iron Gwazi")}')
    print(f'get_accel_filename("VelociCoaster") → {get_accel_filename("VelociCoaster")}')
    print(f'get_ratings_name("Zadra") → {get_ratings_name("Zadra")}')
    print(f'get_ratings_name("VelociCoas") → {get_ratings_name("VelociCoas")}')
