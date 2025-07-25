import googlemaps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import json
import requests
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HyderabadTrafficAnalyzer:
    """
    Enhanced Real-time traffic analysis system optimized for GitHub Actions
    """

    def __init__(self, api_key: str):
        self.gmaps = googlemaps.Client(key=api_key)
        self.data_collection = []
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.start_time = None
        self.master_filename = "hyderabad_traffic_master.csv"

        # Define Urban and Suburban study areas for Hyderabad
        self.urban_routes = {
            'tank_bund_abids': {
                'origin': 'Tank Bund, Hyderabad',
                'destination': 'Abids, Hyderabad',
                'description': 'Central Business District'
            },
            'banjara_jubilee': {
                'origin': 'Banjara Hills, Hyderabad',
                'destination': 'Jubilee Hills, Hyderabad',
                'description': 'Prime Commercial Area'
            },
            'charminar_sultan': {
                'origin': 'Charminar, Hyderabad',
                'destination': 'Sultan Bazaar, Hyderabad',
                'description': 'Old City Dense Area'
            },
            'hitec_cyberabad': {
                'origin': 'Hitec City, Hyderabad',
                'destination': 'Cyberabad, Hyderabad',
                'description': 'Tech Hub Core'
            }
        }

        self.suburban_routes = {
            'gachibowli_miyapur': {
                'origin': 'Gachibowli, Hyderabad',
                'destination': 'Miyapur, Hyderabad',
                'description': 'Tech Corridor Suburban'
            },
            'kompally_nizampet': {
                'origin': 'Kompally, Hyderabad',
                'destination': 'Nizampet, Hyderabad',
                'description': 'Residential Suburban'
            },
            'shamirpet_medchal': {
                'origin': 'Shamirpet, Hyderabad',
                'destination': 'Medchal, Hyderabad',
                'description': 'Outer Suburban'
            },
            'orr_eastern': {
                'origin': 'Outer Ring Road, Kompally',
                'destination': 'Outer Ring Road, LB Nagar',
                'description': 'Suburban Connectivity'
            }
        }

    def load_existing_data(self) -> pd.DataFrame:
        """Load existing data from master file if it exists"""
        if os.path.exists(self.master_filename):
            try:
                df = pd.read_csv(self.master_filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"📂 Loaded {len(df)} existing records from {self.master_filename}")
                return df
            except Exception as e:
                print(f"⚠️ Error loading existing data: {e}")
                return pd.DataFrame()
        else:
            print(f"📂 No existing data file found, starting fresh")
            return pd.DataFrame()

    def test_api_connection(self):
        """Test if the Google Maps API is working properly"""
        print("🔧 Testing Google Maps API connection...")
        try:
            # Test with a simple route
            result = self.gmaps.distance_matrix(
                origins=['Hyderabad'],
                destinations=['Secunderabad'],
                mode="driving"
            )

            if result['status'] == 'OK':
                print("✅ API connection successful!")
                print(f"   Status: {result['status']}")
                if result['rows'][0]['elements'][0]['status'] == 'OK':
                    duration = result['rows'][0]['elements'][0]['duration']['text']
                    distance = result['rows'][0]['elements'][0]['distance']['text']
                    print(f"   Test route - Distance: {distance}, Duration: {duration}")
                    return True
                else:
                    print(f"❌ Route status: {result['rows'][0]['elements'][0]['status']}")
                    return False
            else:
                print(f"❌ API Status: {result['status']}")
                return False

        except Exception as e:
            print(f"❌ API connection failed: {str(e)}")
            return False

    def collect_single_measurement(self, routes_to_monitor: dict) -> List[dict]:
        """Collect data for one measurement cycle and return the data points"""
        timestamp = datetime.now()
        print(f"\n⏰ Data collection at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        cycle_data = []

        for route_id, route_info in routes_to_monitor.items():
            try:
                print(f"  📍 Processing {route_id}...", end=' ')

                # Make API call
                result = self.gmaps.distance_matrix(
                    origins=[route_info['origin']],
                    destinations=[route_info['destination']],
                    mode="driving",
                    departure_time=timestamp,
                    traffic_model="best_guess",
                    avoid=None
                )

                self.api_calls_made += 1

                if result['rows'][0]['elements'][0]['status'] == 'OK':
                    element = result['rows'][0]['elements'][0]

                    # Extract traffic data
                    duration_in_traffic = element.get('duration_in_traffic', element['duration'])
                    normal_duration = element['duration']
                    distance = element['distance']

                    # Calculate traffic metrics
                    delay_factor = duration_in_traffic['value'] / normal_duration['value']
                    avg_speed = distance['value'] / duration_in_traffic['value'] * 3.6  # km/h

                    data_point = {
                        'timestamp': timestamp,
                        'route_id': route_id,
                        'route_type': 'urban' if route_id.startswith('urban_') else 'suburban',
                        'route_name': route_info['description'],
                        'origin': route_info['origin'],
                        'destination': route_info['destination'],
                        'distance_m': distance['value'],
                        'distance_km': distance['value'] / 1000,
                        'normal_duration_s': normal_duration['value'],
                        'traffic_duration_s': duration_in_traffic['value'],
                        'delay_factor': delay_factor,
                        'delay_minutes': (duration_in_traffic['value'] - normal_duration['value']) / 60,
                        'avg_speed_kmh': avg_speed,
                        'hour': timestamp.hour,
                        'day_of_week': timestamp.strftime('%A'),
                        'is_weekend': timestamp.weekday() >= 5,
                        'congestion_level': self._classify_congestion(delay_factor)
                    }

                    cycle_data.append(data_point)
                    self.successful_calls += 1

                    print(f"✅ {delay_factor:.2f}x delay, {avg_speed:.1f} km/h")

                else:
                    print(f"❌ {result['rows'][0]['elements'][0]['status']}")
                    self.failed_calls += 1

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                self.failed_calls += 1

            # Small delay between API calls
            time.sleep(1)

        # Show cycle summary
        if cycle_data:
            avg_delay = np.mean([d['delay_factor'] for d in cycle_data])
            avg_speed = np.mean([d['avg_speed_kmh'] for d in cycle_data])
            print(f"📊 Collected {len(cycle_data)} data points")
            print(f"   Average delay factor: {avg_delay:.2f}x")
            print(f"   Average speed: {avg_speed:.1f} km/h")

        return cycle_data

    def collect_and_append_data(self, route_type: str = 'both') -> bool:
        """
        Collect traffic data for current time and append to master file
        Optimized for GitHub Actions single execution
        """
        # Test API first
        if not self.test_api_connection():
            print("❌ Cannot proceed with data collection - API test failed")
            return False

        print(f"\n🚗 Starting single traffic data collection for Hyderabad...")
        
        self.start_time = datetime.now()
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0

        # Prepare routes
        routes_to_monitor = {}
        if route_type in ['urban', 'both']:
            routes_to_monitor.update({f"urban_{k}": v for k, v in self.urban_routes.items()})
        if route_type in ['suburban', 'both']:
            routes_to_monitor.update({f"suburban_{k}": v for k, v in self.suburban_routes.items()})

        print(f"🛣️  Monitoring {len(routes_to_monitor)} routes:")
        for route_id, route_info in routes_to_monitor.items():
            print(f"   - {route_id}: {route_info['description']}")

        # Collect current data
        new_data = self.collect_single_measurement(routes_to_monitor)
        
        if not new_data:
            print("❌ No data collected in this cycle")
            return False

        # Load existing data
        existing_df = self.load_existing_data()
        
        # Create DataFrame from new data
        new_df = pd.DataFrame(new_data)
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        
        # Combine with existing data
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates based on timestamp and route_id
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'route_id'], keep='last')
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        else:
            combined_df = new_df

        # Save to master file
        try:
            combined_df.to_csv(self.master_filename, index=False)
            print(f"💾 Data saved to '{self.master_filename}' ({len(combined_df)} total records)")
            print(f"   Added {len(new_data)} new records")
            
            # Show recent statistics
            if len(combined_df) > 0:
                print(f"📊 Current statistics:")
                print(f"   - Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
                print(f"   - Average delay factor: {combined_df['delay_factor'].mean():.2f}x")
                print(f"   - Average speed: {combined_df['avg_speed_kmh'].mean():.1f} km/h")
                
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False

    def _classify_congestion(self, delay_factor: float) -> str:
        """Classify congestion level based on delay factor"""
        if delay_factor < 1.2:
            return 'Light'
        elif delay_factor < 1.5:
            return 'Moderate'
        elif delay_factor < 2.0:
            return 'Heavy'
        else:
            return 'Severe'

def main():
    """
    Main function optimized for GitHub Actions
    """
    # Get API key from environment variable
    API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not API_KEY:
        print("❌ Error: GOOGLE_MAPS_API_KEY environment variable not found")
        print("Please set the API key as a GitHub secret")
        return False

    print("🔑 API key loaded from environment")
    
    # Initialize analyzer
    analyzer = HyderabadTrafficAnalyzer(API_KEY)

    # Collect and append data
    success = analyzer.collect_and_append_data(route_type='both')
    
    if success:
        print("✅ Data collection completed successfully!")
        return True
    else:
        print("❌ Data collection failed!")
        return False

if __name__ == "__main__":
    success = main()
    # Exit with appropriate code for GitHub Actions
    exit(0 if success else 1)