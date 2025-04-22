# Media Player App

A modern Android media player application that allows users to play media files from their device and create custom playlists by referencing media without moving the files.

## Features

- Browse and play media files from device storage
- Create custom playlists
- Add media to playlists without moving the actual files
- Background playback with notification controls
- Modern Material Design UI
- Support for Android 14 (SDK 34)

## Technical Details

### Architecture

The app follows the MVVM (Model-View-ViewModel) architecture pattern with the following components:

- **Model**: Room database for storing playlists and media references
- **View**: UI components (Activities, Fragments, RecyclerViews)
- **ViewModel**: Manages UI-related data and business logic

### Libraries and Components

- **AndroidX**: Core libraries for modern Android development
- **Room**: SQLite object mapping for local database
- **LiveData & ViewModel**: For reactive UI updates and lifecycle management
- **MediaPlayer**: For audio playback
- **Navigation Component**: For managing fragment navigation
- **RecyclerView**: For efficient lists
- **Coroutines**: For asynchronous operations

## Permissions

The app requires the following permissions:

- `READ_MEDIA_AUDIO` (Android 13+) / `READ_EXTERNAL_STORAGE` (Android 12 and below): To access media files
- `FOREGROUND_SERVICE`: For background media playback
- `FOREGROUND_SERVICE_MEDIA_PLAYBACK`: For media playback foreground service (Android 14+)

## Setup and Installation

1. Clone the repository
2. Open the project in Android Studio
3. Connect an Android device or use an emulator
4. Build and run the application

## Usage

1. Grant storage permissions when prompted
2. Browse your media files in the "Media" tab
3. Create playlists in the "Playlists" tab
4. Long-press or tap the menu icon on media items to add them to playlists
5. Enjoy your music with the built-in media player
