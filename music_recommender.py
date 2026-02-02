"""
Music Recommendation Module
Maps detected emotions to appropriate music recommendations
"""

import random

class MusicRecommender:
    """Class for emotion-based music recommendations"""
    
    def __init__(self, use_spotify=False):
        """Initialize the music recommender"""
        self.use_spotify = use_spotify
        
        # Emotion to music genre/mood mapping
        self.emotion_music_map = {
            "Happy": {
                "genres": ["Pop", "Dance", "Upbeat", "Electronic"],
                "playlists": [
                    "Happy Hits",
                    "Feel Good Pop",
                    "Dance Party",
                    "Upbeat Vibes",
                    "Good Mood"
                ],
                "songs": [
                    "Happy - Pharrell Williams",
                    "Can't Stop the Feeling - Justin Timberlake",
                    "Walking on Sunshine - Katrina and the Waves",
                    "Good Vibrations - The Beach Boys",
                    "Don't Stop Me Now - Queen"
                ]
            },
            "Sad": {
                "genres": ["Acoustic", "Piano", "Soft Rock", "Blues"],
                "playlists": [
                    "Sad Songs",
                    "Melancholy Piano",
                    "Rainy Day",
                    "Emotional Ballads",
                    "Life Sucks"
                ],
                "songs": [
                    "Someone Like You - Adele",
                    "The Scientist - Coldplay",
                    "Hurt - Johnny Cash",
                    "Mad World - Gary Jules",
                    "Tears in Heaven - Eric Clapton"
                ]
            },
            "Angry": {
                "genres": ["Rock", "Metal", "Hard Rock", "Punk"],
                "playlists": [
                    "Rage Beats",
                    "Heavy Metal",
                    "Rock Classics",
                    "Angry Music",
                    "Workout Motivation"
                ],
                "songs": [
                    "Break Stuff - Limp Bizkit",
                    "Killing in the Name - Rage Against the Machine",
                    "Bodies - Drowning Pool",
                    "Enter Sandman - Metallica",
                    "Smells Like Teen Spirit - Nirvana"
                ]
            },
            "Fear": {
                "genres": ["Ambient", "Calm", "Meditation", "Classical"],
                "playlists": [
                    "Calm & Peaceful",
                    "Meditation Music",
                    "Relaxing Ambient",
                    "Stress Relief",
                    "Deep Focus"
                ],
                "songs": [
                    "Weightless - Marconi Union",
                    "Clair de Lune - Debussy",
                    "Gymnopédie No.1 - Erik Satie",
                    "Spiegel im Spiegel - Arvo Pärt",
                    "Ambient 1 - Brian Eno"
                ]
            },
            "Surprise": {
                "genres": ["Upbeat", "Electronic", "Indie", "Alternative"],
                "playlists": [
                    "Unexpected Hits",
                    "Indie Mix",
                    "Electronic Vibes",
                    "Alternative Rock",
                    "Discover Weekly"
                ],
                "songs": [
                    "Electric Feel - MGMT",
                    "Pumped Up Kicks - Foster the People",
                    "Take On Me - a-ha",
                    "Mr. Blue Sky - Electric Light Orchestra",
                    "September - Earth, Wind & Fire"
                ]
            },
            "Neutral": {
                "genres": ["Chill", "Lo-fi", "Jazz", "Instrumental"],
                "playlists": [
                    "Chill Vibes",
                    "Lo-fi Beats",
                    "Jazz Classics",
                    "Study Music",
                    "Background Music"
                ],
                "songs": [
                    "Lofi Hip Hop Radio",
                    "Blue in Green - Miles Davis",
                    "Autumn Leaves - Bill Evans",
                    "Nujabes - Feather",
                    "Café Music Mix"
                ]
            },
            "Disgust": {
                "genres": ["Instrumental", "Classical", "Ambient"],
                "playlists": [
                    "Peaceful Instrumentals",
                    "Classical Essentials",
                    "Calm Piano",
                    "Soothing Sounds",
                    "Relaxation"
                ],
                "songs": [
                    "Canon in D - Pachelbel",
                    "Air on the G String - Bach",
                    "Moonlight Sonata - Beethoven",
                    "River Flows in You - Yiruma",
                    "Comptine d'un autre été - Yann Tiersen"
                ]
            }
        }
    
    def recommend_music(self, emotion):
        """
        Get music recommendations based on detected emotion
        Returns: dictionary with genres, playlists, and songs
        """
        if emotion not in self.emotion_music_map:
            emotion = "Neutral"
        
        recommendations = self.emotion_music_map[emotion]
        
        # Shuffle for variety
        shuffled_songs = recommendations["songs"].copy()
        random.shuffle(shuffled_songs)
        
        return {
            "emotion": emotion,
            "genres": recommendations["genres"],
            "playlists": recommendations["playlists"][:3],  # Top 3 playlists
            "songs": shuffled_songs[:5]  # Top 5 songs
        }
    
    def get_playlist_url(self, emotion):
        """Get Spotify playlist URL for the emotion (placeholder)"""
        # This would integrate with Spotify API in production
        spotify_playlists = {
            "Happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
            "Sad": "https://open.spotify.com/playlist/37i9dQZF1DX3YSRoSdA634",
            "Angry": "https://open.spotify.com/playlist/37i9dQZF1DX1tyCD9QhIWF",
            "Fear": "https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u",
            "Surprise": "https://open.spotify.com/playlist/37i9dQZF1DX4dyzvuaRJ0n",
            "Neutral": "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ",
            "Disgust": "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO"
        }
        return spotify_playlists.get(emotion, spotify_playlists["Neutral"])

if __name__ == "__main__":
    # Test the recommender
    recommender = MusicRecommender()
    
    print("🎵 Music Recommendation System Test\n")
    
    for emotion in ["Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral", "Disgust"]:
        print(f"\n{'='*50}")
        print(f"Emotion: {emotion}")
        print('='*50)
        
        recommendations = recommender.recommend_music(emotion)
        
        print(f"\n🎸 Genres: {', '.join(recommendations['genres'])}")
        print(f"\n📻 Playlists:")
        for playlist in recommendations['playlists']:
            print(f"  • {playlist}")
        
        print(f"\n🎵 Recommended Songs:")
        for song in recommendations['songs']:
            print(f"  • {song}")

