package com.example.mediaplayerapp.data

import androidx.room.Database
import androidx.room.RoomDatabase
import com.example.mediaplayerapp.data.dao.PlaylistDao
import com.example.mediaplayerapp.data.dao.PlaylistMediaCrossRefDao
import com.example.mediaplayerapp.data.entity.MediaItem
import com.example.mediaplayerapp.data.entity.Playlist
import com.example.mediaplayerapp.data.entity.PlaylistMediaCrossRef

@Database(
    entities = [
        MediaItem::class,
        Playlist::class,
        PlaylistMediaCrossRef::class
    ],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun playlistDao(): PlaylistDao
    abstract fun playlistMediaCrossRefDao(): PlaylistMediaCrossRefDao
} 