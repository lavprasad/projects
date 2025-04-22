package com.example.mediaplayerapp.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Entity representing a media item from the device storage
 */
@Entity(tableName = "media_items")
data class MediaItem(
    @PrimaryKey
    val mediaId: String,
    val title: String,
    val artist: String?,
    val album: String?,
    val duration: Long,
    val uri: String,
    val artworkUri: String?
) 