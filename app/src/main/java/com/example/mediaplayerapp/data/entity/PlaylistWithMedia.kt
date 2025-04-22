package com.example.mediaplayerapp.data.entity

import androidx.room.Embedded
import androidx.room.Junction
import androidx.room.Relation

/**
 * Represents a playlist with its associated media items
 */
data class PlaylistWithMedia(
    @Embedded
    val playlist: Playlist,
    
    @Relation(
        parentColumn = "playlistId",
        entityColumn = "mediaId",
        associateBy = Junction(
            value = PlaylistMediaCrossRef::class,
            parentColumn = "playlistId",
            entityColumn = "mediaId"
        )
    )
    val mediaItems: List<MediaItem>
) 