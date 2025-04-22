package com.example.mediaplayerapp.data.dao

import androidx.lifecycle.LiveData
import androidx.room.*
import com.example.mediaplayerapp.data.entity.MediaItem
import com.example.mediaplayerapp.data.entity.Playlist
import com.example.mediaplayerapp.data.entity.PlaylistWithMedia

@Dao
interface PlaylistDao {
    @Insert
    suspend fun insertPlaylist(playlist: Playlist): Long

    @Update
    suspend fun updatePlaylist(playlist: Playlist)

    @Delete
    suspend fun deletePlaylist(playlist: Playlist)

    @Query("SELECT * FROM playlists ORDER BY createdAt DESC")
    fun getAllPlaylists(): LiveData<List<Playlist>>

    @Query("SELECT * FROM playlists WHERE playlistId = :playlistId")
    suspend fun getPlaylistById(playlistId: Long): Playlist?

    @Transaction
    @Query("SELECT * FROM playlists WHERE playlistId = :playlistId")
    fun getPlaylistWithMedia(playlistId: Long): LiveData<PlaylistWithMedia>
    
    @Insert
    suspend fun insertMediaItem(mediaItem: MediaItem)
    
    @Query("SELECT * FROM media_items WHERE mediaId = :mediaId")
    suspend fun getMediaItemById(mediaId: String): MediaItem?
} 