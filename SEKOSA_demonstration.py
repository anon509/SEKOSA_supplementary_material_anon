"""
SEKOSA Demonstration Script
============================

This script demonstrates how SEKOSA assesses robot behaviors under varying
environmental conditions, reproducing the demonstration provided in our paper.

Requirements:
    pip install typedb-driver matplotlib numpy

Usage:
    python SEKOSA_demo_script.py
"""

import sys
from typedb.driver import TypeDB, SessionType, TransactionType, TypeDBOptions
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional


# Configuration
DATABASE_NAME = "sekosa_old"
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
                        $light isa LightIntensity, has description "LightIntensity";
                        $light has average_lumen $old_intensity;
                    delete
                        $light has $old_intensity;
                    insert
                        $light has average_lumen {light_intensity};
                """)
                
                # Update ambient noise
                tx.query.update(f"""
                    match
                        $noise isa AmbientNoise, has description "AmbientNoise";
                        $noise has average_decibel $old_noise;
                    delete
                        $noise has $old_noise;
                    insert
                        $noise has average_decibel {ambient_noise};
                """)
                
                # Update room size (room diagonal)
                tx.query.update(f"""
                    match
                        $room isa RoomSize;
                        $room has diagonal $old_size;
                    delete
                        $room has $old_size;
                    insert
                        $room has diagonal {room_size};
                """)
                
                # Update victim pose error distances based on room size
                # For acoustic search: error = room_size
                tx.query.update(f"""
                    match
                        $vicpose isa VictimPose, has description "Estimated victim location auditive search";
                        $vicpose has average_error_distance_acoustic $old_error;
                    delete
                        $vicpose has $old_error;
                    insert
                        $vicpose has average_error_distance_acoustic {room_size};
                """)
                
                # For fast search: error = 0.75 * room_size
                fast_error = 0.75 * room_size
                tx.query.update(f"""
                    match
                        $vicpose isa VictimPose, has description "Estimated victim location fast search";
                        $vicpose has average_error_distance_fast $old_error;
                    delete
                        $vicpose has $old_error;
                    insert
                        $vicpose has average_error_distance_fast {fast_error};
                """)
                
                # For thorough search: error = waypoint_distance + robot_pose_error
                # Assuming waypoint_distance scales with room (e.g., 0.75 * room_size)
                # and robot_pose_error is fixed at 0.25
                thorough_error = fast_error + 0.25
                tx.query.update(f"""
                    match
                        $vicpose isa VictimPose, has description "Estimated victim location thorough search";
                        $vicpose has average_error_distance_thorough $old_error;
                    delete
                        $vicpose has $old_error;
                    insert
                        $vicpose has average_error_distance_thorough {thorough_error};
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
                        $light isa LightIntensity, has average_lumen $intensity;
                    get $light, $intensity;
                """)
                for answer in result:
                    conditions['light_intensity'] = answer.get("intensity")._value()
                    conditions['light variable'] = answer.get("light")
                
                # Ambient Noise
                result = tx.query.get("""
                    match
                        $noise isa AmbientNoise, has average_decibel $decibel;
                    get $noise, $decibel;
                """)
                for answer in result:
                    conditions['ambient_noise'] = answer.get("decibel")._value()
                    conditions['noise variable'] = answer.get("noise")
                
                # Room Size
                result = tx.query.get("""
                    match
                        $room isa RoomSize, has diagonal $size;
                    get $room, $size;
                """)
                for answer in result:
                    conditions['room_size'] = answer.get("size")._value()
                    conditions['room variable'] = answer.get("room")
        
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
                
                # Query for behavior with processing_requirement
                if "Thorough" in actual_behavior_name:
                    query = f"""
                        match
                            $behavior isa Behaviour, has name "{actual_behavior_name}";
                            $pr (petitioner: $behavior, output: $detection, output: $vicpose) isa processing_requirement;
                            $detection isa Detection, has average_change_correct_detection $p_human;
                            $vicpose isa VictimPose, has average_error_distance_thorough $d_l;
                        get $p_human, $d_l;
                    """

                elif "Fast" in actual_behavior_name:
                    query = f"""
                        match
                            $behavior isa Behaviour, has name "{actual_behavior_name}";
                            $pr (petitioner: $behavior, output: $detection, output: $vicpose) isa processing_requirement;
                            $detection isa Detection, has average_change_correct_detection $p_human;
                            $vicpose isa VictimPose, has average_error_distance_fast $d_l;
                        get $p_human, $d_l;
                    """
                
                elif "Acoustic" in actual_behavior_name:
                    query = f"""
                        match
                            $behavior isa Behaviour, has name "{actual_behavior_name}";
                            $pr (petitioner: $behavior, output: $detection, output: $vicpose) isa processing_requirement;
                            $detection isa Detection, has average_change_correct_detection $p_human;
                            $vicpose isa VictimPose, has average_error_distance_acoustic $d_l;
                        get $p_human, $d_l;
                    """
                #print("Querying performance for behavior:", behavior_name, "with query:\n", query)
                result = tx.query.get(query)
                answers = list(result)

                if len(answers) == 0:
                    #print("No performance data found for behavior:", behavior_name)
                    return None
                
                for answer in answers:
                    #print("Answer:", answer)
                    p_human = answer.get("p_human")._value()
                    d_l = answer.get("d_l")._value()
                    #print(f"Behavior: {behavior_name}, P(human): {p_human}, D_l: {d_l}")
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
            #print(f"Assessing behavior: {behavior} with metrics: {metrics}")
            #print(f"P(human): {metrics['p_human'] if metrics else 'N/A'}, D_l: {metrics['d_l'] if metrics else 'N/A'}")
            #print(f"Type of P(human): {type(metrics['p_human'].as_double()) if metrics else 'N/A'}, Type of D_l: {type(metrics['d_l']) if metrics else 'N/A'}")
            if metrics is not None:
                p_human = metrics['p_human'].as_double()
                d_l = metrics['d_l'].as_double()
                if p_human >= p_human_threshold and d_l <= d_l_threshold:
                    viable_behaviors.append((behavior, p_human, d_l))
        
        # Priority: thorough > fast > audio (based on paper)
        priority = {"thorough-search": 0, "fast-search": 1, "audio-search": 2}
        
        viable_behaviors.sort(key=lambda x: priority.get(x[0], 999))
        
        if viable_behaviors:
            selected_behavior = viable_behaviors[0][0]
            #print(f"Selected behavior: {selected_behavior} with P(human)={viable_behaviors[0][1]}, D_l={viable_behaviors[0][2]}")
            return selected_behavior
        else:
            #print("No viable behavior found under current conditions.")
            return None
    
def run_test_conditions(demonstrator: SEKOSADemonstrator, conditions: Tuple[float, float, float], behaviors: List[str]=["thorough-search", "fast-search", "audio-search"]):
    """
    Runs a test to evaluate the behaviors under the conditions provided as (light, noise, room_size).
    First, it updates the environmental conditions in the knowledge base.
    Then, it checks the pre-conditions for inference of each of the behaviors provided.
    Finally, it queries and prints the P(human) and D_l values for each behavior.
    """
    behaviors = ["thorough-search"]
    light_intensity, ambient_noise, room_size = conditions
    print("\n" + "="*60)
    print(f"Running Test Conditions: Light={light_intensity} lm, Noise={ambient_noise} dB, Room Size={room_size} m")
    print("Evaluating behaviors:", behaviors)
    print("="*60)
    
    demonstrator.update_environmental_conditions(light_intensity, ambient_noise, room_size)
    actual_conditions = demonstrator.get_environmental_conditions()
    print("Updated Environmental Conditions:", actual_conditions)
    
    for behavior in behaviors:
        if behavior == "audio-search":
            preconditions = '''
                    $beh isa Behaviour, has name "Acoustic Search";
                    $proc1 (input:$noise, executor:$SpeechToText, output:$detection) isa processing;
                    $noise isa AmbientNoise;
                    $SpeechToText isa SpeechToText;
                    $detection isa Creation, has average_change_correct_detection $quality_change_correct_detection;
                    $proc2 (input:$roomdiag, executor:$NLP, output:$vicpose) isa processing;
                    $roomdiag isa RoomSize;
                    $NLP isa NLP;
                    $vicpose isa VictimPose, has average_error_distance_acoustic $quality_error_distance_acoustic;
                    $minimal_allowed_quality_change_correct_detection isa minimal_allowed_change_correct_detection;
                    $minimal_allowed_quality_error_distance isa minimal_allowed_error_distance;
                    $quality_change_correct_detection > $minimal_allowed_quality_change_correct_detection;
                    $quality_error_distance_acoustic > $minimal_allowed_quality_error_distance;
                    '''
            performance_fetches = '''
                    get $quality_change_correct_detection, $quality_error_distance_acoustic;
                    '''
        elif behavior == "thorough-search":
            preconditions = '''
                    $beh isa Behaviour, has name "Visual Thorough Search";
                    $proc1 (input:$light, executor:$cam, output:$detection) isa processing;
                    $light isa LightIntensity;
                    $detection isa Creation, has average_change_correct_detection $quality_change_correct_detection;
                    $proc2 (input:$robpos, executor:$track, output:$vicpose) isa processing;
                    $proc3 (input:$detection, executor:$track, output:$vicpose) isa processing;
                    $track isa ObjectTracking;
                    $robpos isa RobotPose;
                    $vicpose isa VictimPose, has average_error_distance_thorough $quality_error_distance_thorough;
                    $minimal_allowed_quality_change_correct_detection isa minimal_allowed_change_correct_detection;
                    $minimal_allowed_quality_error_distance isa minimal_allowed_error_distance;
                    $quality_change_correct_detection > $minimal_allowed_quality_change_correct_detection;
                    $quality_error_distance_thorough > $minimal_allowed_quality_error_distance;
                    '''
            # prepare debug queries that check which preconditions fail
            debug_queries = [
                '$light isa LightIntensity;',
                '$noise isa AmbientNoise;',
                '$roomdiag isa RoomSize;',
                '$robpos isa RobotPose;',
                '$proc1 (input:$light, executor:$cam, output:$detection) isa processing;'
                '$proc2 (input:$robpos, executor:$track, output:$vicpose) isa processing;',
                '$proc3 (input:$detection, executor:$track, output:$vicpose) isa processing;',
                '$vicpose isa VictimPose, has average_error_distance_thorough $quality_error_distance_thorough;',
                '$detection isa Creation, has average_change_correct_detection $quality_change_correct_detection;',
                '$quality_change_correct_detection > $minimal_allowed_quality_change_correct_detection;',
                '$quality_error_distance_thorough > $minimal_allowed_quality_error_distance;'
            ]
            performance_fetches = '''
                    get $quality_change_correct_detection, $quality_error_distance_thorough;
                    '''
        elif behavior == "fast-search":
            preconditions = '''
                    $beh isa Behaviour, has name "Visual Fast Search";
                    $proc1 (input:$light, executor:$cam, output:$detection) isa processing;
                    $light isa LightIntensity;
                    $detection isa Creation, has average_change_correct_detection $quality_change_correct_detection;
                    $proc2 (input:$robpos, executor:$track, output:$vicpose) isa processing;
                    $proc3 (input:$detection, executor:$track, output:$vicpose) isa processing;
                    $track isa ObjectTracking;
                    $robpos isa RobotPose;
                    $vicpose isa VictimPose, has average_error_distance_fast $quality_error_distance_fast;
                    $minimal_allowed_quality_change_correct_detection isa minimal_allowed_change_correct_detection;
                    $minimal_allowed_quality_error_distance isa minimal_allowed_error_distance;
                    $quality_change_correct_detection > $minimal_allowed_quality_change_correct_detection;
                    $quality_error_distance_fast > $minimal_allowed_quality_error_distance;
                    '''
            performance_fetches = '''
                    get $quality_change_correct_detection, $quality_error_distance_fast;
                    '''
            
        with demonstrator.driver.session(demonstrator.database_name, SessionType.DATA) as session:
            with session.transaction(TransactionType.READ,
                                    options=TypeDBOptions(infer=True)) as tx:
                # check the precondition query first
                precondition_query = f"""
                match
                {preconditions}
                get;
                """
                false_conditions = True
                try:
                    print("Precondition Query:\n", precondition_query)
                    answers = list(tx.query.get(precondition_query))
                    if len(answers) > 0:
                        print(f"Precondition query returned {len(answers)} answers.")
                        false_conditions = False
                    else:
                        print(f"Precondition query returned no answers.")
                        false_conditions = True

                except Exception as e:
                    print(f"Error querying preconditions for behavior '{behavior}': {e}")
                    
                if not false_conditions:
                    print(f"All preconditions for behavior '{behavior}' are satisfied.")
                    print("Now querying performance values...")
                    performance_query = """
                    match
                    {preconditions}
                    {performance_fetches}
                    """
                    performance_query = performance_query.format(
                        preconditions=preconditions,
                        performance_fetches=performance_fetches
                    )
                    try:
                        print("Performance Query:\n", performance_query)
                        answers = list(tx.query.get(performance_query))
                        if len(answers) > 0:
                            for answer in answers:
                                p_human = answer.get("quality_change_correct_detection")._value()
                                if behavior == "thorough-search":
                                    d_l = answer.get("quality_error_distance_thorough")._value()
                                elif behavior == "fast-search":
                                    d_l = answer.get("quality_error_distance_fast")._value()
                                elif behavior == "audio-search":
                                    d_l = answer.get("quality_error_distance_acoustic")._value()
                                
                                print(f"Behavior: {behavior}")
                                print(f"  P(human): {p_human}")
                                print(f"  D_l: {d_l} m")
                        else:
                            print(f"No answer...")
                    except Exception as e:
                        print(f"Error querying performance for behavior '{behavior}': {e}")
                else:
                    print(f"Behavior: {behavior} is not viable under current conditions.")
                    continue
                
                        
        
        performance = demonstrator.get_behavior_performance(behavior)
        if performance is not None:
            print(f"Behavior: {behavior}")
            print(f"  P(human): {performance['p_human']}")
            print(f"  D_l: {performance['d_l']} m")
        else:
            print(f"Behavior: {behavior} is not viable under current conditions.")


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
    light_range = np.linspace(25, 250, 10)  # 25-250 lm
    noise_range = np.linspace(75, 95, 10)  # 75-95 dB
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
    light_range = np.linspace(25, 250, 10)  # 25-250 lm
    room_range = np.linspace(2, 7, 10)     # 2-7 m diagonal
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

def investigate_results(results: np.ndarray, range_option: int):
    """
    Turns a results ndarray into a more human readable format for investigation.
    Prints the results in the format:
    Light Intensity: X lm, Ambient Noise: Y dB -> Selected Behavior: Z (P(human)=A, D_l=B)
    """
    if range_option == 1:
        x_range = np.linspace(25, 250, 10)  # 25-250 lm
        y_range = np.linspace(75, 95, 10)  # 75-95 dB
    elif range_option == 2:
        x_range = np.linspace(25, 250, 10)  # 25-250 lm
        y_range = np.linspace(2, 7, 10)     # 2-7 m diagonal
    
    behavior_map = {
        0: "No Viable Behavior",
        1: "Audio Search",
        2: "Fast Search",
        3: "Thorough Search"
    }
    
    for i, noise in enumerate(y_range):
        for j, light in enumerate(x_range):
            behavior_code = results[i, j]
            behavior_name = behavior_map.get(behavior_code, "Unknown")
            if range_option == 1:
                print(f"Light Intensity: {light:.1f} lm, Ambient Noise: {noise:.1f} dB -> Selected Behavior: {behavior_name}")
            elif range_option == 2:
                print(f"Light Intensity: {light:.1f} lm, Room Diagonal: {noise:.1f} m -> Selected Behavior: {behavior_name}")


def main():
    """Main demonstration routine."""
    print("="*60)
    print("SEKOSA Demonstration Script")
    print("Reproducing results from the paper")
    print("="*60)
    
    demonstrator = SEKOSADemonstrator(DATABASE_NAME, SERVER_ADDRESS)
    
    try:
        demonstrator.connect()

        # test conditions: [25, 75, 2.4], [85, 62, 3.5]
        #run_test_conditions(demonstrator, (50, 75, 2.4))
        #run_test_conditions(demonstrator, (150, 85, 2.4))
        #run_test_conditions(demonstrator, (85, 62, 3.5))
        
        # Uncomment when working:
        results1 = run_demonstration_1(demonstrator)
        results2 = run_demonstration_2(demonstrator)
        plot_results(results1, results2)

        # Optional: Investigate results in human-readable format
        #print("\nInvestigating Demonstration 1 results:")
        #investigate_results(results1, range_option=1)
        #print("\nInvestigating Demonstration 2 results:")
        #investigate_results(results2, range_option=2)
        
    finally:
        demonstrator.close()


if __name__ == "__main__":
    main()
