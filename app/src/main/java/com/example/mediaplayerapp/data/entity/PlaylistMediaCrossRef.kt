package com.example.mediaplayerapp.data.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index

/**
 * Cross-reference entity to handle the many-to-many relationship
 * between playlists and media items
 */
@Entity(
    tableName = "playlist_media_cross_ref",
    primaryKeys = ["playlistId", "mediaId"],
    foreignKeys = [
        ForeignKey(
            entity = Playlist::class,
            parentColumns = ["playlistId"],
            childColumns = ["playlistId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index("playlistId"), 
        Index("mediaId")
    ]
)
data class PlaylistMediaCrossRef(
    val playlistId: Long,
    val mediaId: String,
    val position: Int // For ordering media within a playlist
) 