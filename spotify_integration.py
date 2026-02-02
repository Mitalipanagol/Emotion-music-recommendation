"""
Optional Spotify API Integration
Provides real music playback using Spotify API
"""

import os
from typing import List, Dict

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    print("⚠️  Spotipy not installed. Install with: pip install spotipy")

class SpotifyMusicRecommender:
    """Enhanced music recommender with Spotify integration"""
    
    def __init__(self, client_id=None, client_secret=None):
        """
        Initialize Spotify recommender
        
        To use this:
        1. Go to https://developer.spotify.com/dashboard
        2. Create an app
        3. Get your Client ID and Client Secret
        4. Set environment variables or pass them here
        """
        
        if not SPOTIFY_AVAILABLE:
            raise ImportError("Spotipy is not installed. Run: pip install spotipy")
        
        # Get credentials from environment or parameters
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET environment variables or pass them to the constructor."
            )
        
        # Initialize Spotify client
        auth_manager = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # Emotion to search query mapping
        self.emotion_queries = {
            "Happy": ["happy", "upbeat", "feel good", "cheerful"],
            "Sad": ["sad", "melancholy", "emotional", "heartbreak"],
            "Angry": ["angry", "rage", "metal", "hard rock"],
            "Fear": ["calm", "peaceful", "meditation", "relaxing"],
            "Surprise": ["unexpected", "indie", "alternative", "eclectic"],
            "Neutral": ["chill", "lofi", "ambient", "background"],
            "Disgust": ["classical", "instrumental", "peaceful"]
        }
    
    def search_playlists(self, emotion: str, limit: int = 5) -> List[Dict]:
        """Search for playlists based on emotion"""
        queries = self.emotion_queries.get(emotion, ["music"])
        
        playlists = []
        for query in queries[:2]:  # Use first 2 queries
            try:
                results = self.sp.search(q=query, type='playlist', limit=limit)
                for item in results['playlists']['items']:
                    playlists.append({
                        'name': item['name'],
                        'url': item['external_urls']['spotify'],
                        'description': item.get('description', ''),
                        'tracks': item['tracks']['total']
                    })
            except Exception as e:
                print(f"Error searching playlists: {e}")
        
        return playlists[:limit]
    
    def search_tracks(self, emotion: str, limit: int = 10) -> List[Dict]:
        """Search for tracks based on emotion"""
        queries = self.emotion_queries.get(emotion, ["music"])
        
        tracks = []
        for query in queries[:2]:
            try:
                results = self.sp.search(q=query, type='track', limit=limit)
                for item in results['tracks']['items']:
                    tracks.append({
                        'name': item['name'],
                        'artist': ', '.join([artist['name'] for artist in item['artists']]),
                        'url': item['external_urls']['spotify'],
                        'preview_url': item.get('preview_url'),
                        'album': item['album']['name']
                    })
            except Exception as e:
                print(f"Error searching tracks: {e}")
        
        return tracks[:limit]
    
    def get_recommendations_by_emotion(self, emotion: str) -> Dict:
        """Get comprehensive recommendations for an emotion"""
        return {
            'emotion': emotion,
            'playlists': self.search_playlists(emotion, limit=5),
            'tracks': self.search_tracks(emotion, limit=10)
        }
    
    def get_featured_playlists(self) -> List[Dict]:
        """Get Spotify's featured playlists"""
        try:
            results = self.sp.featured_playlists(limit=10)
            playlists = []
            for item in results['playlists']['items']:
                playlists.append({
                    'name': item['name'],
                    'url': item['external_urls']['spotify'],
                    'description': item.get('description', '')
                })
            return playlists
        except Exception as e:
            print(f"Error getting featured playlists: {e}")
            return []

def setup_spotify_credentials():
    """Interactive setup for Spotify credentials"""
    print("="*70)
    print("🎵 SPOTIFY API SETUP")
    print("="*70)
    
    print("\n📋 Steps to get Spotify API credentials:")
    print("   1. Go to https://developer.spotify.com/dashboard")
    print("   2. Log in with your Spotify account")
    print("   3. Click 'Create an App'")
    print("   4. Fill in app name and description")
    print("   5. Copy your Client ID and Client Secret")
    
    print("\n" + "="*70)
    
    client_id = input("Enter your Spotify Client ID: ").strip()
    client_secret = input("Enter your Spotify Client Secret: ").strip()
    
    if client_id and client_secret:
        print("\n💾 Saving credentials to environment variables...")
        print("\nAdd these lines to your environment:")
        print(f"   export SPOTIFY_CLIENT_ID='{client_id}'")
        print(f"   export SPOTIFY_CLIENT_SECRET='{client_secret}'")
        
        print("\nOr on Windows:")
        print(f"   set SPOTIFY_CLIENT_ID={client_id}")
        print(f"   set SPOTIFY_CLIENT_SECRET={client_secret}")
        
        # Test the credentials
        try:
            recommender = SpotifyMusicRecommender(client_id, client_secret)
            print("\n✅ Credentials are valid!")
            
            # Test search
            print("\n🔍 Testing search...")
            playlists = recommender.search_playlists("Happy", limit=3)
            print(f"   Found {len(playlists)} playlists")
            
            print("\n🎉 Spotify integration is ready!")
            return True
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
    else:
        print("\n❌ Invalid credentials")
        return False

if __name__ == "__main__":
    if not SPOTIFY_AVAILABLE:
        print("❌ Spotipy not installed.")
        print("   Install with: pip install spotipy")
    else:
        print("🎵 Spotify Integration Test\n")
        
        # Check for existing credentials
        if os.getenv('SPOTIFY_CLIENT_ID') and os.getenv('SPOTIFY_CLIENT_SECRET'):
            print("✅ Found Spotify credentials in environment\n")
            
            try:
                recommender = SpotifyMusicRecommender()
                
                # Test recommendations
                emotion = "Happy"
                print(f"Testing recommendations for emotion: {emotion}\n")
                
                recommendations = recommender.get_recommendations_by_emotion(emotion)
                
                print("📻 Playlists:")
                for playlist in recommendations['playlists'][:3]:
                    print(f"   • {playlist['name']}")
                    print(f"     {playlist['url']}")
                
                print("\n🎵 Tracks:")
                for track in recommendations['tracks'][:5]:
                    print(f"   • {track['name']} - {track['artist']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("⚠️  Spotify credentials not found in environment\n")
            setup_spotify_credentials()

