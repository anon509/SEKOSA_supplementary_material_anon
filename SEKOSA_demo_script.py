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
from typedb.driver import TypeDB, SessionType, TransactionType
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
    """Demonstrates SEKOSA behavior assessment under varying conditions."""
    
    def __init__(self, address: str = SERVER_ADDRESS, database: str = DATABASE_NAME):
        """Initialize connection to TypeDB.
        
        Args:
            address: TypeDB server address
            database: Database name
        """
        self.address = address
        self.database = database
        self.driver = None
        
    def connect(self):
        """Connect to TypeDB server."""
        try:
            self.driver = TypeDB.core_driver(self.address)
            print(f"✓ Connected to TypeDB at {self.address}")
            
            # Check if database exists
            if not self.driver.databases.contains(self.database):
                print(f"✗ Database '{self.database}' does not exist!")
                print(f"  Please create it first with the schema and data files.")
                sys.exit(1)
            else:
                print(f"✓ Database '{self.database}' found")
                
        except Exception as e:
            print(f"✗ Failed to connect to TypeDB: {e}")
            sys.exit(1)
    
    def close(self):
        """Close TypeDB connection."""
        if self.driver:
            self.driver.close()
            print("✓ Connection closed")
    
    def update_environment(self, light: float, noise: float, room_size: float):
        """Update environmental conditions in the knowledge base.
        
        Args:
            light: Light intensity in Lumen
            noise: Ambient noise in dB
            room_size: Room diagonal in meters
        """
        with self.driver.session(self.database, SessionType.DATA) as session:
            with session.transaction(TransactionType.WRITE) as tx:
                
                # Update light intensity
                query_delete_light = """
                    match
                    $light isa LightIntensity, has average_lumen $lumen, 
                           has description "LightIntensity";
                    delete
                    $light has $lumen;
                """
                tx.query(query_delete_light)
                
                query_insert_light = f"""
                    match
                    $light isa LightIntensity, has description "LightIntensity";
                    insert
                    $light has average_lumen {light};
                """
                tx.query(query_insert_light)
                
                # Update ambient noise
                query_delete_noise = """
                    match
                    $noise isa AmbientNoise, has average_decibel $db,
                           has description "AmbientNoise";
                    delete
                    $noise has $db;
                """
                tx.query(query_delete_noise)
                
                query_insert_noise = f"""
                    match
                    $noise isa AmbientNoise, has description "AmbientNoise";
                    insert
                    $noise has average_decibel {noise};
                """
                tx.query(query_insert_noise)
                
                # Update room size
                query_delete_room = """
                    match
                    $room isa RoomSize, has diagonal $diag;
                    delete
                    $room has $diag;
                """
                tx.query(query_delete_room)
                
                query_insert_room = f"""
                    match
                    $room isa RoomSize;
                    insert
                    $room has diagonal {room_size};
                """
                tx.query(query_insert_room)
                
                # Update victim pose error distances based on room size
                # Fast search: 3/4 of room diagonal
                # Thorough search: waypoint distance (3m) + robot pose error (0.25m) 
                #                  scaled by room size
                # Acoustic search: room diagonal
                
                victim_pose_fast = 0.75 * room_size
                victim_pose_thorough = 3.0 + 0.25  # Simplified from paper
                victim_pose_acoustic = room_size
                
                # Update fast search victim pose error
                query_update_fast = f"""
                    match
                    $vp isa average_error_distance_fast;
                    delete
                    $vp isa average_error_distance_fast;
                    insert
                    {victim_pose_fast} isa average_error_distance_fast;
                """
                tx.query(query_update_fast)
                
                # Update thorough search victim pose error
                query_update_thorough = f"""
                    match
                    $vp isa average_error_distance_thorough;
                    delete
                    $vp isa average_error_distance_thorough;
                    insert
                    {victim_pose_thorough} isa average_error_distance_thorough;
                """
                tx.query(query_update_thorough)
                
                # Update acoustic search victim pose error
                query_update_acoustic = f"""
                    match
                    $vp isa average_error_distance_acoustic;
                    delete
                    $vp isa average_error_distance_acoustic;
                    insert
                    {victim_pose_acoustic} isa average_error_distance_acoustic;
                """
                tx.query(query_update_acoustic)
                
                tx.commit()
    
    def assess_behaviors(self) -> Dict[str, Optional[Tuple[float, float]]]:
        """Assess all behaviors under current environmental conditions.
        
        Returns:
            Dictionary mapping behavior names to (P(human), D_l) tuples,
            or None if behavior is not available.
        """
        behaviors = {
            "Visual Fast Search": None,
            "Visual Thorough Search": None,
            "Acoustic Search": None
        }
        
        with self.driver.session(self.database, SessionType.DATA) as session:
            with session.transaction(TransactionType.READ) as tx:
                
                # Check Visual Fast Search
                query_fast = """
                    match
                    $beh isa Behaviour, has name "Visual Fast Search";
                    $pr (processing_requirement: need: $proc, petitioner: $beh, 
                         output: $detection, output: $vicpose) isa processing_requirement;
                    $detection isa Detection, has average_change_correct_detection $quality_det;
                    $vicpose isa VictimPose, has average_error_distance_fast $error_dist;
                    get $quality_det, $error_dist;
                """
                result = list(tx.query(query_fast).resolve())
                if result:
                    ans = result[0]
                    p_human = ans.get("quality_det").as_attribute().get_value()
                    d_l = ans.get("error_dist").as_attribute().get_value()
                    behaviors["Visual Fast Search"] = (p_human, d_l)
                
                # Check Visual Thorough Search
                query_thorough = """
                    match
                    $beh isa Behaviour, has name "Visual Thorough Search";
                    $pr (processing_requirement: need: $proc, petitioner: $beh,
                         output: $detection, output: $vicpose) isa processing_requirement;
                    $detection isa Detection, has average_change_correct_detection $quality_det;
                    $vicpose isa VictimPose, has average_error_distance_thorough $error_dist;
                    get $quality_det, $error_dist;
                """
                result = list(tx.query(query_thorough).resolve())
                if result:
                    ans = result[0]
                    p_human = ans.get("quality_det").as_attribute().get_value()
                    d_l = ans.get("error_dist").as_attribute().get_value()
                    behaviors["Visual Thorough Search"] = (p_human, d_l)
                
                # Check Acoustic Search
                query_acoustic = """
                    match
                    $beh isa Behaviour, has name "Acoustic Search";
                    $pr (processing_requirement: need: $proc, petitioner: $beh,
                         output: $detection, output: $vicpose) isa processing_requirement;
                    $detection isa Detection, has average_change_correct_detection $quality_det;
                    $vicpose isa VictimPose, has average_error_distance_acoustic $error_dist;
                    get $quality_det, $error_dist;
                """
                result = list(tx.query(query_acoustic).resolve())
                if result:
                    ans = result[0]
                    p_human = ans.get("quality_det").as_attribute().get_value()
                    d_l = ans.get("error_dist").as_attribute().get_value()
                    behaviors["Acoustic Search"] = (p_human, d_l)
        
        return behaviors
    
    def select_best_behavior(self, behaviors: Dict[str, Optional[Tuple[float, float]]]) -> Optional[str]:
        """Select the best viable behavior based on performance thresholds.
        
        Args:
            behaviors: Dictionary of behavior assessments
            
        Returns:
            Name of selected behavior, or None if no viable behavior exists
        """
        viable_behaviors = {}
        
        for name, metrics in behaviors.items():
            if metrics is None:
                continue
            
            p_human, d_l = metrics
            
            # Check if behavior meets performance thresholds
            if p_human >= P_HUMAN_THRESHOLD and d_l <= D_L_THRESHOLD:
                viable_behaviors[name] = (p_human, d_l)
        
        if not viable_behaviors:
            return None
        
        # Selection priority: Thorough > Fast > Acoustic
        # (Thorough search provides best quality when viable)
        if "Visual Thorough Search" in viable_behaviors:
            return "Visual Thorough Search"
        elif "Visual Fast Search" in viable_behaviors:
            return "Visual Fast Search"
        elif "Acoustic Search" in viable_behaviors:
            return "Acoustic Search"
        
        return None
    
    def demonstrate_scenario(self, light: float, noise: float, room_size: float):
        """Demonstrate behavior selection for a specific scenario.
        
        Args:
            light: Light intensity in Lumen
            noise: Ambient noise in dB
            room_size: Room diagonal in meters
        """
        print(f"\n{'='*70}")
        print(f"Scenario: Light={light} Lumen, Noise={noise} dB, Room={room_size}m")
        print(f"{'='*70}")
        
        # Update environment
        print("Updating environmental conditions...")
        self.update_environment(light, noise, room_size)
        
        # Assess behaviors
        print("Assessing behaviors...")
        behaviors = self.assess_behaviors()
        
        # Display results
        print("\nBehavior Assessment:")
        for name, metrics in behaviors.items():
            if metrics is None:
                print(f"  ✗ {name}: NOT AVAILABLE")
            else:
                p_human, d_l = metrics
                viable = "✓" if (p_human >= P_HUMAN_THRESHOLD and d_l <= D_L_THRESHOLD) else "✗"
                print(f"  {viable} {name}: P(human)={p_human:.2f}, D_l={d_l:.2f}m")
        
        # Select best behavior
        selected = self.select_best_behavior(behaviors)
        if selected:
            print(f"\n→ SELECTED: {selected}")
        else:
            print(f"\n→ NO VIABLE BEHAVIOR")
    
    def generate_figure_13(self):
        """Generate plots similar to Figure 13 from the paper.
        
        This creates two heatmaps showing behavior selection under varying conditions.
        """
        print("\n" + "="*70)
        print("Generating Figure 13: Behavior Selection Under Varying Conditions")
        print("="*70)
        
        # Define ranges (based on paper)
        light_range = np.linspace(20, 150, 14)
        noise_range = np.linspace(70, 100, 16)
        room_range = np.linspace(2, 6, 17)
        
        # Fixed values for each subplot
        fixed_room = 2.4  # meters
        fixed_noise = 85  # dB
        
        # Create result matrices
        # Encoding: 0=no behavior, 1=acoustic, 2=fast, 3=thorough
        result_noise_light = np.zeros((len(noise_range), len(light_range)))
        result_room_light = np.zeros((len(room_range), len(light_range)))
        
        behavior_encoding = {
            None: 0,
            "Acoustic Search": 1,
            "Visual Fast Search": 2,
            "Visual Thorough Search": 3
        }
        
        print("\nComputing behavior selection map (this may take a few minutes)...")
        
        # Subplot 1: Varying noise and light (fixed room size)
        print(f"\nSubplot 1: Room size fixed at {fixed_room}m")
        total = len(noise_range) * len(light_range)
        count = 0
        for i, noise in enumerate(noise_range):
            for j, light in enumerate(light_range):
                count += 1
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
                
                self.update_environment(light, noise, fixed_room)
                behaviors = self.assess_behaviors()
                selected = self.select_best_behavior(behaviors)
                result_noise_light[i, j] = behavior_encoding[selected]
        
        # Subplot 2: Varying room size and light (fixed noise)
        print(f"\nSubplot 2: Ambient noise fixed at {fixed_noise} dB")
        total = len(room_range) * len(light_range)
        count = 0
        for i, room in enumerate(room_range):
            for j, light in enumerate(light_range):
                count += 1
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
                
                self.update_environment(light, fixed_noise, room)
                behaviors = self.assess_behaviors()
                selected = self.select_best_behavior(behaviors)
                result_room_light[i, j] = behavior_encoding[selected]
        
        # Create figure
        print("\nGenerating plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Define colors: black, white, dark grey, light grey
        from matplotlib.colors import ListedColormap
        colors = ['black', 'white', 'darkgray', 'lightgray']
        cmap = ListedColormap(colors)
        
        # Subplot 1
        im1 = ax1.imshow(result_noise_light, cmap=cmap, aspect='auto',
                         extent=[light_range[0], light_range[-1],
                                noise_range[0], noise_range[-1]],
                         origin='lower', vmin=0, vmax=3)
        ax1.set_xlabel('Light Intensity [Lumen]')
        ax1.set_ylabel('Ambient Noise [dB]')
        ax1.set_title(f'Behavior Selection (Room Size = {fixed_room}m)')
        
        # Subplot 2
        im2 = ax2.imshow(result_room_light, cmap=cmap, aspect='auto',
                         extent=[light_range[0], light_range[-1],
                                room_range[0], room_range[-1]],
                         origin='lower', vmin=0, vmax=3)
        ax2.set_xlabel('Light Intensity [Lumen]')
        ax2.set_ylabel('Room Diagonal [m]')
        ax2.set_title(f'Behavior Selection (Ambient Noise = {fixed_noise} dB)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='No Viable Behavior'),
            Patch(facecolor='white', label='Acoustic Search'),
            Patch(facecolor='darkgray', label='Visual Fast Search'),
            Patch(facecolor='lightgray', label='Visual Thorough Search')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                  bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        
        # Save figure
        output_file = 'sekosa_figure13_reproduction.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved as '{output_file}'")
        
        plt.show()


def main():
    """Main demonstration function."""
    print("="*70)
    print("SEKOSA Demonstration: Behavior Assessment Under Varying Conditions")
    print("="*70)
    
    demo = SEKOSADemonstrator()
    
    try:
        # Connect to TypeDB
        demo.connect()
        
        # Demonstrate specific scenarios from the paper
        print("\n" + "="*70)
        print("PART 1: Individual Scenario Demonstrations")
        print("="*70)
        
        # Scenario 1: Good lighting, moderate noise, small room
        # Expected: Visual Fast Search
        demo.demonstrate_scenario(light=130, noise=85, room_size=2.4)
        
        # Scenario 2: Low lighting, moderate noise, small room
        # Expected: Acoustic Search
        demo.demonstrate_scenario(light=70, noise=85, room_size=2.4)
        
        # Scenario 3: Very low lighting, high noise, small room
        # Expected: No viable behavior
        demo.demonstrate_scenario(light=40, noise=95, room_size=2.4)
        
        # Scenario 4: Good lighting, moderate noise, large room
        # Expected: Visual Thorough Search
        demo.demonstrate_scenario(light=130, noise=85, room_size=5.0)
        
        # Generate full Figure 13
        print("\n" + "="*70)
        print("PART 2: Generate Complete Figure 13")
        print("="*70)
        
        response = input("\nGenerate complete Figure 13? (This will take several minutes) [y/N]: ")
        if response.lower() == 'y':
            demo.generate_figure_13()
        else:
            print("Skipping Figure 13 generation.")
        
        print("\n" + "="*70)
        print("Demonstration Complete!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
