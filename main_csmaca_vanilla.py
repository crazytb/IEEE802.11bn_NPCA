import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class IEEE80211Simulation:
    def __init__(self):
        # IEEE 802.11ax parameters (in microseconds)
        self.slot_time = 9  # µs
        self.sifs = 16  # µs
        self.difs = 34  # µs (DIFS = SIFS + 2 * slot_time)
        self.cwmin = 16
        self.cwmax = 1024
        
        # Available channels from the 6 GHz band only
        self.available_channels = {
            # U-NII-5
            1: {"width": 20, "frequency": 5955},
            5: {"width": 20, "frequency": 5975},
            9: {"width": 20, "frequency": 5995},
            13: {"width": 20, "frequency": 6015},
            17: {"width": 20, "frequency": 6035},
            21: {"width": 20, "frequency": 6055},
            25: {"width": 20, "frequency": 6075},
            29: {"width": 20, "frequency": 6095},
            33: {"width": 20, "frequency": 6115},
            37: {"width": 20, "frequency": 6135},
            41: {"width": 20, "frequency": 6155},
            45: {"width": 20, "frequency": 6175},
            49: {"width": 20, "frequency": 6195},
            53: {"width": 20, "frequency": 6215},
            57: {"width": 20, "frequency": 6235},
            61: {"width": 20, "frequency": 6255},
            65: {"width": 20, "frequency": 6275},
            69: {"width": 20, "frequency": 6295},
            73: {"width": 20, "frequency": 6315},
            77: {"width": 20, "frequency": 6335},
            81: {"width": 20, "frequency": 6355},
            85: {"width": 20, "frequency": 6375},
            89: {"width": 20, "frequency": 6395},
            93: {"width": 20, "frequency": 6415},
            
            # U-NII-6
            97: {"width": 20, "frequency": 6435},
            101: {"width": 20, "frequency": 6455},
            105: {"width": 20, "frequency": 6475},
            109: {"width": 20, "frequency": 6495},
            113: {"width": 20, "frequency": 6515},
            117: {"width": 20, "frequency": 6535},
            121: {"width": 20, "frequency": 6555},
            125: {"width": 20, "frequency": 6575},
            129: {"width": 20, "frequency": 6595},
            133: {"width": 20, "frequency": 6615},
            137: {"width": 20, "frequency": 6635},
            141: {"width": 20, "frequency": 6655},
            145: {"width": 20, "frequency": 6675},
            149: {"width": 20, "frequency": 6695},
            153: {"width": 20, "frequency": 6715},
            157: {"width": 20, "frequency": 6735},
            161: {"width": 20, "frequency": 6755},
            165: {"width": 20, "frequency": 6775},
            169: {"width": 20, "frequency": 6795},
            173: {"width": 20, "frequency": 6815},
            177: {"width": 20, "frequency": 6835},
            181: {"width": 20, "frequency": 6855},
            185: {"width": 20, "frequency": 6875},
            
            # U-NII-7
            189: {"width": 20, "frequency": 6895},
            193: {"width": 20, "frequency": 6915},
            197: {"width": 20, "frequency": 6935},
            201: {"width": 20, "frequency": 6955},
            205: {"width": 20, "frequency": 6975},
            209: {"width": 20, "frequency": 6995},
            213: {"width": 20, "frequency": 7015},
            217: {"width": 20, "frequency": 7035},
            221: {"width": 20, "frequency": 7055},
            225: {"width": 20, "frequency": 7075},
            
            # U-NII-8
            229: {"width": 20, "frequency": 7095},
            233: {"width": 20, "frequency": 7115},
        }
        
        # Maps for wider channel bonding - 6 GHz band only
        self.channel_bonding_40MHz = {
            # 6 GHz band
            1: [1, 5],
            9: [9, 13],
            17: [17, 21],
            25: [25, 29],
            33: [33, 37],
            41: [41, 45],
            49: [49, 53],
            57: [57, 61],
            65: [65, 69],
            73: [73, 77],
            81: [81, 85],
            89: [89, 93],
            97: [97, 101],
            105: [105, 109],
            113: [113, 117],
            121: [121, 125],
            129: [129, 133],
            137: [137, 141],
            145: [145, 149],
            153: [153, 157],
            161: [161, 165],
            169: [169, 173],
            177: [177, 181],
            185: [185, 189],
            193: [193, 197],
            201: [201, 205],
            209: [209, 213],
            217: [217, 221],
            225: [225, 229]
        }
        
        self.channel_bonding_80MHz = {
            # 6 GHz band
            1: [1, 5, 9, 13],
            17: [17, 21, 25, 29],
            33: [33, 37, 41, 45],
            49: [49, 53, 57, 61],
            65: [65, 69, 73, 77],
            81: [81, 85, 89, 93],
            97: [97, 101, 105, 109],
            113: [113, 117, 121, 125],
            129: [129, 133, 137, 141],
            145: [145, 149, 153, 157],
            161: [161, 165, 169, 173],
            177: [177, 181, 185, 189],
            193: [193, 197, 201, 205],
            209: [209, 213, 217, 221]
        }
        
        self.channel_bonding_160MHz = {
            # 6 GHz band
            1: [1, 5, 9, 13, 17, 21, 25, 29],
            33: [33, 37, 41, 45, 49, 53, 57, 61],
            65: [65, 69, 73, 77, 81, 85, 89, 93],
            97: [97, 101, 105, 109, 113, 117, 121, 125],
            129: [129, 133, 137, 141, 145, 149, 153, 157],
            161: [161, 165, 169, 173, 177, 181, 185, 189],
            193: [193, 197, 201, 205, 209, 213, 217, 221]
        }
        
        # State variables
        self.stations = {}
        self.channel_status = {ch: {"busy": False, "backoff_counter": -1} for ch in self.available_channels}
        self.simulation_time = 0
        self.max_simulation_time = 1000000  # 1 second in µs
        self.collision_count = 0
        self.successful_transmissions = 0
        self.events = []
        
        # Statistics
        self.throughput_by_channel = defaultdict(int)
        self.channel_usage = defaultdict(int)
        self.backoff_stats = []

    def setup_stations(self, station_counts):
        """
        Set up stations with their primary channels.
        
        Args:
            station_counts: dict mapping channel numbers to number of stations
        """
        station_id = 0
        for channel, count in station_counts.items():
            if channel not in self.available_channels:
                print(f"Warning: Channel {channel} not available, skipping")
                continue
                
            for i in range(count):
                self.stations[station_id] = {
                    "primary_channel": channel,
                    "state": "IDLE",  # IDLE, BACKOFF, TRANSMITTING
                    "backoff_counter": -1,
                    "retry_count": 0,
                    "cw": self.cwmin,
                    "channel_width": 20,  # Default to 20MHz
                    "actual_tx_width": 20,  # Actual transmission width after adaptation
                    "data_queued": True,  # Assume always has data to send
                    "transmission_time": 0
                }
                station_id += 1
    
    def setup_channel_width(self, station_id, width):
        """
        Set up channel width for a station (20, 40, 80, 160 MHz).
        
        Args:
            station_id: ID of the station
            width: Channel width in MHz (20, 40, 80, 160)
        """
        if station_id not in self.stations:
            print(f"Station {station_id} does not exist")
            return
            
        self.stations[station_id]["channel_width"] = width
    
    def are_channels_idle(self, channels, check_time):
        """Check if all specified channels are idle at the given time."""
        for ch in channels:
            # Skip if channel not available
            if ch not in self.available_channels:
                continue
                
            # Check channel status
            if self.channel_status[ch]["busy"] and self.channel_status[ch]["backoff_counter"] > check_time:
                return False
        return True
    
    def check_channel_overlap(self, channels_to_check):
        """
        Check if the specified channels overlap with any currently busy channels.
        Returns True if there's overlap, False otherwise.
        """
        for ch in channels_to_check:
            if ch in self.available_channels and self.channel_status[ch]["busy"]:
                return True
        return False
    
    def get_non_overlapping_channels(self, primary_ch, desired_width):
        """
        Get the widest possible channel bonding up to desired_width without overlapping busy channels.
        Returns a list of channels to use.
        """
        # Start with just the primary channel
        if primary_ch not in self.available_channels:
            return [primary_ch]
            
        # If only 20MHz is desired or primary channel is busy, just return primary
        if desired_width == 20 or self.channel_status[primary_ch]["busy"]:
            return [primary_ch]
            
        # Try increasingly wider channel bondings until we hit an overlap or reach desired width
        if desired_width >= 40:
            # Get potential 40MHz channels
            channels_40 = None
            for base_ch, channels in self.channel_bonding_40MHz.items():
                if primary_ch in channels:
                    channels_40 = channels
                    break
                    
            if not channels_40:
                # Create 40MHz bonding if not found in predefined bonding
                if primary_ch % 2 == 1:  # Odd channel
                    next_ch = primary_ch + 4
                    if next_ch in self.available_channels:
                        channels_40 = [primary_ch, next_ch]
                    else:
                        prev_ch = primary_ch - 4
                        if prev_ch in self.available_channels:
                            channels_40 = [prev_ch, primary_ch]
                else:  # Even channel
                    prev_ch = primary_ch - 4
                    if prev_ch in self.available_channels:
                        channels_40 = [prev_ch, primary_ch]
                    else:
                        next_ch = primary_ch + 4
                        if next_ch in self.available_channels:
                            channels_40 = [primary_ch, next_ch]
            
            # Check if any of the 40MHz channels are busy
            if channels_40 and not self.check_channel_overlap(channels_40):
                # If 40MHz is all we want or all we can get, return it
                if desired_width == 40:
                    return channels_40
                    
                # Try for 80MHz if desired
                if desired_width >= 80:
                    channels_80 = None
                    min_ch_40 = min(channels_40)
                    
                    # Find predefined 80MHz bonding
                    for base_ch, channels in self.channel_bonding_80MHz.items():
                        if primary_ch in channels:
                            channels_80 = channels
                            break
                    
                    if not channels_80:
                        # Try to create 80MHz bonding by finding adjacent 40MHz group
                        if min_ch_40 % 8 == 1:  # First in an 80MHz block
                            next_group_base = min_ch_40 + 4
                            if next_group_base in self.channel_bonding_40MHz:
                                channels_80 = channels_40 + self.channel_bonding_40MHz[next_group_base]
                        elif min_ch_40 % 8 == 5:  # Second in an 80MHz block
                            prev_group_base = min_ch_40 - 4
                            if prev_group_base in self.channel_bonding_40MHz:
                                channels_80 = self.channel_bonding_40MHz[prev_group_base] + channels_40
                    
                    # Check if any of the 80MHz channels are busy
                    if channels_80 and not self.check_channel_overlap(channels_80):
                        # If 80MHz is all we want or all we can get, return it
                        if desired_width == 80:
                            return channels_80
                            
                        # Try for 160MHz if desired
                        if desired_width >= 160:
                            channels_160 = None
                            min_ch_80 = min(channels_80)
                            
                            # Find predefined 160MHz bonding
                            for base_ch, channels in self.channel_bonding_160MHz.items():
                                if primary_ch in channels:
                                    channels_160 = channels
                                    break
                            
                            if not channels_160:
                                # Try to create 160MHz bonding by finding adjacent 80MHz group
                                if min_ch_80 % 16 == 1:  # First in a 160MHz block
                                    next_group_base = min_ch_80 + 8
                                    if next_group_base in self.channel_bonding_80MHz:
                                        channels_160 = channels_80 + self.channel_bonding_80MHz[next_group_base]
                                elif min_ch_80 % 16 == 9:  # Second in a 160MHz block
                                    prev_group_base = min_ch_80 - 8
                                    if prev_group_base in self.channel_bonding_80MHz:
                                        channels_160 = self.channel_bonding_80MHz[prev_group_base] + channels_80
                            
                            # Check if any of the 160MHz channels are busy
                            if channels_160 and not self.check_channel_overlap(channels_160):
                                return channels_160
                            
                            # If 160MHz has overlap, fall back to 80MHz
                            return channels_80
                    
                    # If 80MHz has overlap, fall back to 40MHz
                    return channels_40
            
            # If 40MHz has overlap, fall back to just primary
            return [primary_ch]
        
        # Fallback
        return [primary_ch]
    
    def get_channels_for_station(self, station_id):
        """
        Get all channels a station will use based on its primary channel and width,
        taking into account any channel overlap with busy channels.
        """
        station = self.stations[station_id]
        primary_ch = station["primary_channel"]
        desired_width = station["channel_width"]
        
        # Find the maximum non-overlapping channel bonding up to the desired width
        return self.get_non_overlapping_channels(primary_ch, desired_width)
    
    def start_backoff(self, station_id):
        """Start the backoff procedure for a station."""
        station = self.stations[station_id]
        
        if station["retry_count"] == 0:
            station["cw"] = self.cwmin
        else:
            station["cw"] = min(self.cwmax, station["cw"] * 2)
        
        station["backoff_counter"] = random.randint(0, station["cw"] - 1)
        station["state"] = "BACKOFF"
        
        self.backoff_stats.append(station["backoff_counter"])
    
    def simulate_csma_ca(self):
        """Simulate the CSMA/CA process."""
        self.simulation_time = 0
        last_debug_time = 0
        debug_interval = 100000  # Debug print every 100ms
        
        # Debug print of channel bonding configurations
        print("\nInitial Channel Bonding Configurations:")
        for station_id, station in self.stations.items():
            if station["channel_width"] > 20:
                primary_ch = station["primary_channel"]
                channels = self.get_channels_for_station(station_id)
                print(f"Station {station_id} (Primary Ch {primary_ch}): {station['channel_width']}MHz bonding using channels {channels}")
        
        # Track station channel width adaptations
        channel_width_adaptations = 0
        
        while self.simulation_time < self.max_simulation_time:
            # Periodic debug printing
            if self.simulation_time - last_debug_time > debug_interval:
                print(f"Simulation time: {self.simulation_time/1000:.2f}ms, " +
                     f"Successful transmissions: {self.successful_transmissions}, " +
                     f"Collisions: {self.collision_count}, " +
                     f"Width adaptations: {channel_width_adaptations}")
                last_debug_time = self.simulation_time
                
            # Initialize all stations to start backoff if they're idle with data
            for station_id, station in self.stations.items():
                if station["state"] == "IDLE" and station["data_queued"]:
                    self.start_backoff(station_id)
            
            # Process backoffs
            min_backoff = float('inf')
            min_backoff_stations = []
            
            for station_id, station in self.stations.items():
                if station["state"] == "BACKOFF" and station["backoff_counter"] >= 0:
                    # Get the channels based on dynamic channel bonding with collision avoidance
                    channels = self.get_channels_for_station(station_id)
                    
                    # If we had to reduce channel width due to busy channels, count it
                    if len(channels) * 20 < station["channel_width"]:
                        station["actual_tx_width"] = len(channels) * 20
                        channel_width_adaptations += 1
                    else:
                        station["actual_tx_width"] = station["channel_width"]
                    
                    # Check if all required channels are idle
                    if self.are_channels_idle(channels, self.simulation_time):
                        if station["backoff_counter"] < min_backoff:
                            min_backoff = station["backoff_counter"]
                            min_backoff_stations = [station_id]
                        elif station["backoff_counter"] == min_backoff:
                            min_backoff_stations.append(station_id)
            
            # If no stations are in backoff, advance time
            if not min_backoff_stations:
                self.simulation_time += self.slot_time
                continue
            
            # Advance time to the minimum backoff
            self.simulation_time += min_backoff * self.slot_time
            
            # Check for collision (multiple stations finished backoff simultaneously)
            if len(min_backoff_stations) > 1:
                self.collision_count += 1
                for station_id in min_backoff_stations:
                    self.stations[station_id]["retry_count"] += 1
                    self.start_backoff(station_id)
            else:
                # Successful transmission
                station_id = min_backoff_stations[0]
                station = self.stations[station_id]
                
                # Get the actual channels used for this transmission (may be adapted due to busy channels)
                channels = self.get_channels_for_station(station_id)
                actual_width = station["actual_tx_width"]
                
                # Calculate transmission time (simplified)
                # Using basic data rate assumptions based on channel width
                # 20MHz: 54 Mbps, 40MHz: 108 Mbps, 80MHz: 216 Mbps, 160MHz: 432 Mbps
                # For a 1500 byte packet
                base_tx_time = 222  # 1500 bytes at 54 Mbps
                tx_time = base_tx_time * (20 / actual_width) if actual_width > 0 else base_tx_time
                
                # Mark channels as busy during transmission
                for ch in channels:
                    if ch in self.available_channels:  # Skip unavailable channels
                        self.channel_status[ch]["busy"] = True
                        self.channel_status[ch]["backoff_counter"] = self.simulation_time + tx_time
                        self.channel_usage[ch] += tx_time
                
                # Update station state
                station["state"] = "TRANSMITTING"
                station["transmission_time"] = self.simulation_time + tx_time
                
                # Log event
                self.events.append({
                    "time": self.simulation_time,
                    "type": "transmission",
                    "station": station_id,
                    "channels": channels,
                    "requested_width": station["channel_width"],
                    "actual_width": actual_width
                })
                
                # Update throughput statistics
                primary_ch = station["primary_channel"]
                # Scale throughput by channel width ratio
                throughput_bits = 1500 * 8 * (actual_width / 20)
                self.throughput_by_channel[primary_ch] += throughput_bits
                
                # Update counter
                self.successful_transmissions += 1
                
                # Advance time for transmission + SIFS + ACK
                self.simulation_time += tx_time + self.sifs + 44  # 44µs for ACK
                
                # Update station after transmission
                station["state"] = "IDLE"
                station["retry_count"] = 0
                station["backoff_counter"] = -1
                
                # Reset channel status after transmission
                for ch in channels:
                    if ch in self.available_channels:
                        self.channel_status[ch]["busy"] = False
            
            # Decrement backoff counters for other stations
            for station_id, station in self.stations.items():
                if station["state"] == "BACKOFF" and station["backoff_counter"] > min_backoff:
                    station["backoff_counter"] -= min_backoff
        
        # Calculate final throughput (bits per second)
        throughput_bps = {}
        for channel, bits in self.throughput_by_channel.items():
            throughput_bps[channel] = bits / (self.simulation_time / 1000000)  # convert µs to seconds
            
        # Return simulation results
        return {
            "throughput": throughput_bps,
            "collisions": self.collision_count,
            "successful_tx": self.successful_transmissions,
            "channel_utilization": {ch: usage / self.simulation_time for ch, usage in self.channel_usage.items()},
            "avg_backoff": sum(self.backoff_stats) / len(self.backoff_stats) if self.backoff_stats else 0,
            "width_adaptations": channel_width_adaptations
        }
    
    def visualize_results(self, results):
        """Visualize simulation results."""
        # Plot throughput by channel
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Throughput by channel
        plt.subplot(2, 2, 1)
        channels = sorted(list(results["throughput"].keys()))
        
        # Group channels by U-NII band
        unii5_channels = [ch for ch in channels if 1 <= ch <= 93]
        unii6_channels = [ch for ch in channels if 97 <= ch <= 185]
        unii7_channels = [ch for ch in channels if 189 <= ch <= 225]
        unii8_channels = [ch for ch in channels if ch >= 229]
        
        # Create grouped bar chart
        if unii5_channels:
            unii5_throughputs = [results["throughput"][ch] for ch in unii5_channels]
            plt.bar(unii5_channels, unii5_throughputs, color='blue', alpha=0.7, label='U-NII-5')
        
        if unii6_channels:
            unii6_throughputs = [results["throughput"][ch] for ch in unii6_channels]
            plt.bar(unii6_channels, unii6_throughputs, color='green', alpha=0.7, label='U-NII-6')
            
        if unii7_channels:
            unii7_throughputs = [results["throughput"][ch] for ch in unii7_channels]
            plt.bar(unii7_channels, unii7_throughputs, color='orange', alpha=0.7, label='U-NII-7')
            
        if unii8_channels:
            unii8_throughputs = [results["throughput"][ch] for ch in unii8_channels]
            plt.bar(unii8_channels, unii8_throughputs, color='red', alpha=0.7, label='U-NII-8')
            
        plt.xlabel('Channel')
        plt.ylabel('Throughput (bps)')
        plt.title('Throughput by Channel')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Channel utilization
        plt.subplot(2, 2, 2)
        channels = sorted(list(results["channel_utilization"].keys()))
        
        # Group channels by U-NII band
        unii5_channels = [ch for ch in channels if 1 <= ch <= 93]
        unii6_channels = [ch for ch in channels if 97 <= ch <= 185]
        unii7_channels = [ch for ch in channels if 189 <= ch <= 225]
        unii8_channels = [ch for ch in channels if ch >= 229]
        
        # Create grouped bar chart for utilization
        if unii5_channels:
            unii5_util = [results["channel_utilization"][ch] * 100 for ch in unii5_channels]
            plt.bar(unii5_channels, unii5_util, color='blue', alpha=0.7, label='U-NII-5')
        
        if unii6_channels:
            unii6_util = [results["channel_utilization"][ch] * 100 for ch in unii6_channels]
            plt.bar(unii6_channels, unii6_util, color='green', alpha=0.7, label='U-NII-6')
            
        if unii7_channels:
            unii7_util = [results["channel_utilization"][ch] * 100 for ch in unii7_channels]
            plt.bar(unii7_channels, unii7_util, color='orange', alpha=0.7, label='U-NII-7')
            
        if unii8_channels:
            unii8_util = [results["channel_utilization"][ch] * 100 for ch in unii8_channels]
            plt.bar(unii8_channels, unii8_util, color='red', alpha=0.7, label='U-NII-8')
            
        plt.xlabel('Channel')
        plt.ylabel('Utilization (%)')
        plt.title('Channel Utilization')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Transmission statistics
        plt.subplot(2, 2, 3)
        stats = ['Successful TX', 'Collisions', 'Width Adaptations']
        values = [results["successful_tx"], results["collisions"], results.get("width_adaptations", 0)]
        colors = ['green', 'red', 'orange']
        plt.bar(stats, values, color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Transmission Statistics')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Channel bonding statistics
        plt.subplot(2, 2, 4)
        # Count how many stations use each channel width
        width_counts = {20: 0, 40: 0, 80: 0, 160: 0}
        for station in self.stations.values():
            width_counts[station["channel_width"]] += 1
        
        widths = list(width_counts.keys())
        counts = list(width_counts.values())
        
        plt.bar(widths, counts, color='purple', alpha=0.7)
        plt.xlabel('Channel Width (MHz)')
        plt.ylabel('Number of Stations')
        plt.title('Channel Bonding Usage')
        plt.grid(True, alpha=0.3)
        
        # Add text for average backoff
        plt.figtext(0.5, 0.01, f'Average Backoff Slots: {results["avg_backoff"]:.2f}', 
                   horizontalalignment='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()


def run_simulation_example():
    # Create simulation
    sim = IEEE80211Simulation()
    
    # Set up stations on 6 GHz band only
    station_counts = {
        # 6 GHz band - U-NII-5
        1: 2, 2: 1, 5: 2, 9: 2, 13: 1, 17: 1, 25: 2, 33: 1, 41: 1,
        
        # 6 GHz band - U-NII-6
        97: 2, 113: 2, 129: 1, 145: 1, 161: 1,
        
        # 6 GHz band - U-NII-7
        193: 2, 209: 1,
        
        # 6 GHz band - U-NII-8
        229: 1
    }
    sim.setup_stations(station_counts)
    
    # Set up a scenario to demonstrate secondary channel collision
    # First, set up some stations with channel bonding
    
    # Channel 1 station using 80MHz - this will use channels [1,5,9,13]
    channel_1_station_id = None
    for station_id, station in sim.stations.items():
        if station["primary_channel"] == 1:
            channel_1_station_id = station_id
            break
    if channel_1_station_id is not None:
        sim.setup_channel_width(channel_1_station_id, 80)  # Channel 1 station using 80MHz
    
    # Channel 17 station wanting 160MHz - would overlap with 1's 80MHz if not handled correctly
    # 160MHz would use [1,5,9,13,17,21,25,29] but should be limited to [17,21,25,29]
    channel_17_station_id = None
    for station_id, station in sim.stations.items():
        if station["primary_channel"] == 17:
            channel_17_station_id = station_id
            break
    if channel_17_station_id is not None:
        sim.setup_channel_width(channel_17_station_id, 160)  # Channel 17 station requesting 160MHz
    
    # Channel 2 and 5 examples
    channel_2_station_id = None
    channel_5_station_id = None
    for station_id, station in sim.stations.items():
        if station["primary_channel"] == 2:
            channel_2_station_id = station_id
        elif station["primary_channel"] == 5 and channel_5_station_id is None:  # Get first channel 5 station
            channel_5_station_id = station_id
    
    if channel_2_station_id is not None:
        sim.setup_channel_width(channel_2_station_id, 40)  # Channel 2 station using 40MHz
    
    if channel_5_station_id is not None:
        sim.setup_channel_width(channel_5_station_id, 160)  # Channel 5 station using 160MHz
        
    # Add additional examples from other parts of the spectrum
    # U-NII-6 examples
    for station_id, station in sim.stations.items():
        if station["primary_channel"] == 97:
            sim.setup_channel_width(station_id, 80)  # First channel 97 station using 80MHz
            break
            
    for station_id, station in sim.stations.items():
        if station["primary_channel"] == 113:
            sim.setup_channel_width(station_id, 160)  # First channel 113 station using 160MHz
            break
            
    # U-NII-7 example - will show potential overlap with U-NII-6
    for station_id, station in sim.stations.items():
        if station["primary_channel"] == 193:
            sim.setup_channel_width(station_id, 160)  # Channel 193 station using 160MHz
            break
    
    # Run simulation
    results = sim.simulate_csma_ca()
    
    # Print results
    print("\nSimulation Results:")
    print(f"Total successful transmissions: {results['successful_tx']}")
    print(f"Total collisions: {results['collisions']}")
    print(f"Total channel width adaptations: {results.get('width_adaptations', 0)}")
    print(f"Average backoff slots: {results['avg_backoff']:.2f}")
    print("\nThroughput by channel (Mbps):")
    for ch, throughput in sorted(results["throughput"].items()):
        print(f"  Channel {ch}: {throughput/1000000:.2f} Mbps")
    
    # Visualize results
    sim.visualize_results(results)
    
if __name__ == "__main__":
    run_simulation_example()