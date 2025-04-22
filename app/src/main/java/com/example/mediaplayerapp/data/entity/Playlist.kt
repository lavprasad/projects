package com.example.mediaplayerapp.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Entity representing a user-created playlist
 */
@Entity(tableName = "playlists")
data class Playlist(
    @PrimaryKey(autoGenerate = true)
    val playlistId: Long = 0,
    val name: String,
    val createdAt: Long = System.currentTimeMillis()
) 