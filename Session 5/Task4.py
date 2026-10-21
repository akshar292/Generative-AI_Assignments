def filter_songs_starting_with_s(song_names):
    return [
        song for song in song_names
        if song.lower().startswith("s")
    ]

songs = [
    "Shape of You",
    "Blinding Lights",
    "Senorita",
    "Perfect",
    "Someone Like You",
    "Levitating"
]

filtered_songs = filter_songs_starting_with_s(songs)
print(filtered_songs)