package com.example.mediaplayerapp.data.dao

import androidx.room.*
import com.example.mediaplayerapp.data.entity.PlaylistMediaCrossRef
import kotlinx.coroutines.flow.Flow

@Dao
interface PlaylistMediaCrossRefDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPlaylistMediaCrossRef(crossRef: PlaylistMediaCrossRef)
    
    @Delete
    suspend fun deletePlaylistMediaCrossRef(crossRef: PlaylistMediaCrossRef)
    
    @Query("DELETE FROM playlist_media_cross_ref WHERE playlistId = :playlistId AND mediaId = :mediaId")
    suspend fun removeMediaFromPlaylist(playlistId: Long, mediaId: String)
    
    @Query("DELETE FROM playlist_media_cross_ref WHERE playlistId = :playlistId")
    suspend fun clearPlaylist(playlistId: Long)
    
    @Query("SELECT * FROM playlist_media_cross_ref WHERE playlistId = :playlistId ORDER BY position")
    fun getPlaylistMediaCrossRefs(playlistId: Long): Flow<List<PlaylistMediaCrossRef>>
    
    @Query("SELECT MAX(position) + 1 FROM playlist_media_cross_ref WHERE playlistId = :playlistId")
    suspend fun getNextPositionForPlaylist(playlistId: Long): Int?
} 