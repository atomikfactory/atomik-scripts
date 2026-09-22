#!/usr/bin/python3

from plexapi.server import PlexServer

# === CONFIGURATION ===
PLEX_BASE_URL = 'http://[plex_server_ip]:32400'      # Plex server URL (adjust if not local)
PLEX_TOKEN    = 'YOUR_PLEX_TOKEN'    # Plex token for an admin user
LIBRARY_NAME  = 'YOUR_PERSONAL_LIBRARY' # Name of the library to monitor

# Connect to Plex server
plex = PlexServer(PLEX_BASE_URL, PLEX_TOKEN)

try:
    library = plex.library.section(LIBRARY_NAME)
except Exception as e:
    print(f"Error: Could not find library '{LIBRARY_NAME}'. Check the name. {e}")
    exit(1)

# Fetch all video items in the library
videos = library.all()
reset_count = 0

for video in videos:
    # Check if the item has been played or partially played
    has_offset = getattr(video, 'viewOffset', 0) not in (None, 0)
    was_watched = getattr(video, 'viewCount', 0) > 0
    if has_offset or was_watched:
        # Reset watch status (mark as unwatched)
        try:
            video.markUnwatched()
            reset_count += 1
            print(f"Marked unwatched: {video.title}")
        except Exception as err:
            print(f"Failed to reset '{video.title}': {err}")

print(f"Done. Reset {reset_count} items to unwatched.")
