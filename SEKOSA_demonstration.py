"""
SEKOSA Demonstration Script
============================

This script demonstrates how SEKOSA assesses robot behaviors under varying
environmental conditions, reproducing the demonstration provided in our paper.

Requirements:
    pip install typedb-driver matplotlib numpy

Usage:
    python SEKOSA_demonstration.py
"""

import sys
from typedb.driver import TypeDB, SessionType, TransactionType, TypeDBOptions
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional


# Configuration
DATABASE_NAME = "sekosa"
SERVER_ADDRESS = "localhost:1729"

# Performance thresholds
P_HUMAN_THRESHOLD = 0.5  # Minimum probability of detecting human
D_L_THRESHOLD = 5.0      # Maximum localization uncertainty

class SEKOSADemonstrator:
    """Handles SEKOSA demonstration queries and environmental updates."""
    
    def __init__(self, database_name: str, server_address: str):
        self.database_name = database_name
        self.server_address = server_address
        self.driver = None
        
    def connect(self):
        """Connect to TypeDB server."""
        try:
            self.driver = TypeDB.core_driver(self.server_address)
            print(f"Connected to TypeDB at {self.server_address}")
            
            # Verify database exists
            if not self.driver.databases.contains(self.database_name):
                print(f"Error: Database '{self.database_name}' not found!")
                print("Please create the database and load the schema and data first.")
                sys.exit(1)
                
            print(f"Using database: {self.database_name}")
            
        except Exception as e:
            print(f"Error connecting to TypeDB: {e}")
            sys.exit(1)
    
    def close(self):
        """Close TypeDB connection."""
        if self.driver:
            self.driver.close()
            print("Disconnected from TypeDB")
    
    def update_environmental_conditions(self, light_intensity: float, 
                                   ambient_noise: float, room_size: float):
        """
        Update environmental conditions in the knowledge base.
        
        Args:
            light_intensity: Light intensity in lumens (lm)
            ambient_noise: Ambient noise level in decibels (dB)
            room_size: Room diagonal size in meters (m)
        """
        with self.driver.session(self.database_name, SessionType.DATA) as session:
            with session.transaction(TransactionType.WRITE,
                                    options=TypeDBOptions(infer=True)) as tx:
                # Update light intensity
                tx.query.update(f"""
                    match
                        $light isa Light, has description "Light";
                        $light has light_intensity $old_intensity;
                    delete
                        $light has $old_intensity;
                    insert
                        $light has light_intensity {light_intensity};
                """)
                
                # Update ambient noise
                tx.query.update(f"""
                    match
                        $noise isa Noise, has description "Noise";
                        $noise has noise_level $old_noise;
                    delete
                        $noise has $old_noise;
                    insert
                        $noise has noise_level {ambient_noise};
                """)
                
                # Update room size (room diagonal)
                tx.query.update(f"""
                    match
                        $room isa RoomSize, has description "RoomSize";
                        $room has size $old_size;
                    delete
                        $room has $old_size;
                    insert
                        $room has size {room_size};
                """)
                
                tx.commit()

    def get_environmental_conditions(self) -> Dict[str, float]:
        """
        Retrieve current environmental conditions from the knowledge base.
        
        Returns:
            Dictionary with light_intensity, ambient_noise, room_size
        """
        conditions = {}
        with self.driver.session(self.database_name, SessionType.DATA) as session:
            with session.transaction(TransactionType.READ,
                                    options=TypeDBOptions(infer=True)) as tx:
                # Light Intensity
                result = tx.query.get("""
                    match
                        $light isa Light, has description "Light", has light_intensity $intensity;
                    get $intensity;
                """)
                for answer in result:
                    conditions['light_intensity'] = answer.get("intensity").as_attribute().get_value()
                
                # Ambient Noise
                result = tx.query.get("""
                    match
                        $noise isa Noise, has description "Noise", has noise_level $decibel;
                    get $decibel;
                """)
                for answer in result:
                    conditions['ambient_noise'] = answer.get("decibel").as_attribute().get_value()
                
                # Room Size
                result = tx.query.get("""
                    match
                        $room isa RoomSize, has description "RoomSize", has size $room_size;
                    get $room_size;
                """)
                for answer in result:
                    conditions['room_size'] = answer.get("room_size").as_attribute().get_value()
        
        return conditions
    
    def get_behavior_performance(self, behavior_name: str) -> Optional[Dict[str, float]]:
        """
        Query the performance of a specific behavior under current conditions.
        
        Args:
            behavior_name: Name of the behavior to assess
            
        Returns:
            Dictionary with P(human) and D_l values, or None if behavior not viable
        """
        with self.driver.session(self.database_name, SessionType.DATA) as session:
            with session.transaction(TransactionType.READ,
                                    options=TypeDBOptions(infer=True)) as tx:
                # Map behavior names to match the data
                behavior_map = {
                    "thorough-search": "Visual Thorough Search",
                    "fast-search": "Visual Fast Search",
                    "audio-search": "Acoustic Search"
                }
                
                actual_behavior_name = behavior_map.get(behavior_name, behavior_name)
                
                # Different query for audio search vs visual searches
                if "Acoustic" in actual_behavior_name:
                    query = f"""
                        match
                            $behavior isa Behavior, has name "{actual_behavior_name}";
                            $pr (petitioner: $behavior, output: $detection, output: $vicpose) isa processing_requirement;
                            $detection isa SpeechDetection, has p_human $p_human_val;
                            $vicpose isa VictimPose, has localization_error $d_l_val;
                        get $p_human_val, $d_l_val;
                    """
                else:
                    query = f"""
                        match
                            $behavior isa Behavior, has name "{actual_behavior_name}";
                            $pr (petitioner: $behavior, output: $detection, output: $vicpose) isa processing_requirement;
                            $detection isa Detection, has p_human $p_human_val;
                            $vicpose isa VictimPose, has localization_error $d_l_val;
                        get $p_human_val, $d_l_val;
                    """
                
                result = tx.query.get(query)
                answers = list(result)

                if len(answers) == 0:
                    return None
                
                for answer in answers:
                    p_human = answer.get("p_human_val").as_attribute().get_value()
                    d_l = answer.get("d_l_val").as_attribute().get_value()
                    return {"p_human": p_human, "d_l": d_l}
    
    def assess_all_behaviors(self) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Assess all three search behaviors under current environmental conditions.
        
        Returns:
            Dictionary mapping behavior names to their performance metrics
        """
        behaviors = ["thorough-search", "fast-search", "audio-search"]
        results = {}
        
        for behavior in behaviors:
            results[behavior] = self.get_behavior_performance(behavior)
        
        return results
    
    def select_best_behavior(self, p_human_threshold: float = P_HUMAN_THRESHOLD,
                            d_l_threshold: float = D_L_THRESHOLD) -> Optional[str]:
        """
        Select the best viable behavior meeting performance thresholds.
        
        Args:
            p_human_threshold: Minimum required P(human)
            d_l_threshold: Maximum allowed localization error
            
        Returns:
            Name of selected behavior, or None if no behavior is viable
        """
        performances = self.assess_all_behaviors()
        
        viable_behaviors = []

        for behavior, metrics in performances.items():
            if metrics is not None:
                p_human = metrics['p_human']
                d_l = metrics['d_l']
                if p_human >= p_human_threshold and d_l <= d_l_threshold:
                    combined_score = p_human - (0.1 * d_l)
                    if combined_score < 0:
                        combined_score = 0
                    viable_behaviors.append((behavior, p_human, d_l, combined_score))
        
        # Priority: thorough > fast > audio, to be used if combined scores are equal
        priority = {"thorough-search": 0, "fast-search": 1, "audio-search": 2}
        
        # Select best behavior
        # If a visual search is viable, prefer it over audio search (decide between thorough and fast visual search based on combined score)
        if "thorough-search" in [b[0] for b in viable_behaviors] or "fast-search" in [b[0] for b in viable_behaviors]:
            visual_behaviors = [b for b in viable_behaviors if b[0] in ["thorough-search", "fast-search"]]
            visual_behaviors.sort(key=lambda x: (-x[3], priority[x[0]]))  # Sort by combined score desc, then priority
            return visual_behaviors[0][0]
        elif "audio-search" in [b[0] for b in viable_behaviors]:
            return "audio-search"
        else:
            return None
    

def run_demonstration_1(demonstrator: SEKOSADemonstrator) -> np.ndarray:
    """
    Run Demonstration 1: Vary light intensity and ambient noise (fixed room size).
    
    Returns:
        2D array with behavior selection codes
    """
    print("\n" + "="*60)
    print("Running Demonstration 1: Light Intensity vs Ambient Noise")
    print("Fixed room size: 2.4 m")
    print("="*60)
    
    # Define ranges
    light_range = np.linspace(25, 250, 5)  # 25-250 lm
    noise_range = np.linspace(75, 95, 5)  # 75-95 dB
    room_size = 2.4  # Fixed
    
    # Result matrix: 0=none, 1=audio, 2=fast, 3=thorough
    behavior_map = {
        None: 0,
        "audio-search": 1,
        "fast-search": 2,
        "thorough-search": 3
    }
    
    results = np.zeros((len(noise_range), len(light_range)))
    
    total_iterations = len(noise_range) * len(light_range)
    current_iteration = 0
    
    for i, noise in enumerate(noise_range):
        for j, light in enumerate(light_range):
            # Update conditions
            demonstrator.update_environmental_conditions(light, noise, room_size)
            
            # Select best behavior
            selected = demonstrator.select_best_behavior()
            results[i, j] = behavior_map[selected]
            
            # Progress indicator
            current_iteration += 1
            progress = (current_iteration / total_iterations) * 100
            print(f"Progress: {progress:.1f}%", end="\r")
    
    print(f"Progress: 100.0%")
    print("Demonstration 1 complete!")
    
    return results


def run_demonstration_2(demonstrator: SEKOSADemonstrator) -> np.ndarray:
    """
    Run Demonstration 2: Vary light intensity and room size (fixed ambient noise).
    
    Returns:
        2D array with behavior selection codes
    """
    print("\n" + "="*60)
    print("Running Demonstration 2: Light Intensity vs Room Size")
    print("Fixed ambient noise: 85 dB")
    print("="*60)
    
    # Define ranges
    light_range = np.linspace(25, 250, 5)  # 25-250 lm
    room_range = np.linspace(2, 7, 5)     # 2-7 m diagonal
    ambient_noise = 85.0  # Fixed
    
    # Result matrix: 0=none, 1=audio, 2=fast, 3=thorough
    behavior_map = {
        None: 0,
        "audio-search": 1,
        "fast-search": 2,
        "thorough-search": 3
    }
    
    results = np.zeros((len(room_range), len(light_range)))
    
    total_iterations = len(room_range) * len(light_range)
    current_iteration = 0
    
    for i, room in enumerate(room_range):
        for j, light in enumerate(light_range):
            # Update conditions
            demonstrator.update_environmental_conditions(light, ambient_noise, room)
            
            # Select best behavior
            selected = demonstrator.select_best_behavior()
            results[i, j] = behavior_map[selected]
            
            # Progress indicator
            current_iteration += 1
            progress = (current_iteration / total_iterations) * 100
            print(f"Progress: {progress:.1f}%", end="\r")
    
    print(f"Progress: 100.0%")
    print("Demonstration 2 complete!")
    
    return results


def plot_results(results1: np.ndarray, results2: np.ndarray):
    """
    Plot the demonstration results as heatmaps matching Figure 13 from the paper.
    
    Args:
        results1: Results from demonstration 1 (noise vs light)
        results2: Results from demonstration 2 (room size vs light)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define colormap: 0=black, 1=white, 2=light grey, 3=dark grey
    colors = ['black', 'white', 'lightgrey', 'darkgrey']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot 1: Light Intensity vs Ambient Noise
    ax1.imshow(
        results1,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        origin='upper',
        extent=[25, 250, 95, 75]
    )
    ax1.set_xlabel('Light Intensity (lm)', fontsize=12)
    ax1.set_ylabel('Ambient Noise (dB)', fontsize=12)
    ax1.set_title(
        'Demonstration 1: Fixed Room Size (2.4 m)',
        fontsize=13,
        fontweight='bold'
    )
    ax1.grid(False)
    
    # Plot 2: Light Intensity vs Room Size
    ax2.imshow(
        results2,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        origin='upper',
        extent=[25, 250, 7, 2]
    )
    ax2.set_xlabel('Light Intensity (lm)', fontsize=12)
    ax2.set_ylabel('Room Diagonal (m)', fontsize=12)
    ax2.set_title(
        'Demonstration 2: Fixed Ambient Noise (85 dB)',
        fontsize=13,
        fontweight='bold'
    )
    ax2.grid(False)
    
    # Create shared legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='darkgrey', label='Thorough Search'),
        plt.Rectangle((0, 0), 1, 1, fc='lightgrey', label='Fast Search'),
        plt.Rectangle((0, 0), 1, 1, fc='white', ec='black', label='Audio Search'),
        plt.Rectangle((0, 0), 1, 1, fc='black', label='No Viable Behavior')
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        fontsize=11
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save figure
    plt.savefig(f'{DATABASE_NAME}_demonstration_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {DATABASE_NAME}_demonstration_results.png")
    
    plt.show()


def main():
    """Main demonstration routine."""
    print("="*60)
    print("SEKOSA Demonstration Script")
    print("Reproducing results from the paper")
    print("="*60)
    
    demonstrator = SEKOSADemonstrator(DATABASE_NAME, SERVER_ADDRESS)
    
    try:
        demonstrator.connect()

        results1 = run_demonstration_1(demonstrator)
        results2 = run_demonstration_2(demonstrator)

        plot_results(results1, results2)
        
    finally:
        demonstrator.close()


if __name__ == "__main__":
    main()