package com.example.mediaplayerapp.repository

import androidx.lifecycle.LiveData
import com.example.mediaplayerapp.MediaPlayerApplication
import com.example.mediaplayerapp.data.dao.PlaylistDao
import com.example.mediaplayerapp.data.dao.PlaylistMediaCrossRefDao
import com.example.mediaplayerapp.data.entity.MediaItem
import com.example.mediaplayerapp.data.entity.Playlist
import com.example.mediaplayerapp.data.entity.PlaylistMediaCrossRef
import com.example.mediaplayerapp.data.entity.PlaylistWithMedia
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext

/**
 * Repository for handling playlist operations
 */
class PlaylistRepository {
    private val playlistDao: PlaylistDao = MediaPlayerApplication.database.playlistDao()
    private val playlistMediaCrossRefDao: PlaylistMediaCrossRefDao = MediaPlayerApplication.database.playlistMediaCrossRefDao()
    
    // Playlists
    fun getAllPlaylists(): LiveData<List<Playlist>> = playlistDao.getAllPlaylists()
    
    fun getPlaylistWithMedia(playlistId: Long): LiveData<PlaylistWithMedia> = 
        playlistDao.getPlaylistWithMedia(playlistId)
    
    suspend fun createPlaylist(name: String): Long = withContext(Dispatchers.IO) {
        val playlist = Playlist(name = name)
        return@withContext playlistDao.insertPlaylist(playlist)
    }
    
    suspend fun deletePlaylist(playlist: Playlist) = withContext(Dispatchers.IO) {
        playlistDao.deletePlaylist(playlist)
    }
    
    suspend fun updatePlaylist(playlist: Playlist) = withContext(Dispatchers.IO) {
        playlistDao.updatePlaylist(playlist)
    }
    
    // Media Items
    suspend fun addMediaToPlaylist(playlistId: Long, mediaItem: MediaItem) = withContext(Dispatchers.IO) {
        // Check if the media item already exists in the database, if not insert it
        val existingMediaItem = playlistDao.getMediaItemById(mediaItem.mediaId)
        if (existingMediaItem == null) {
            playlistDao.insertMediaItem(mediaItem)
        }
        
        // Get the next position in the playlist
        val position = playlistMediaCrossRefDao.getNextPositionForPlaylist(playlistId) ?: 0
        
        // Create the cross reference
        val crossRef = PlaylistMediaCrossRef(
            playlistId = playlistId,
            mediaId = mediaItem.mediaId,
            position = position
        )
        
        playlistMediaCrossRefDao.insertPlaylistMediaCrossRef(crossRef)
    }
    
    suspend fun removeMediaFromPlaylist(playlistId: Long, mediaId: String) = withContext(Dispatchers.IO) {
        playlistMediaCrossRefDao.removeMediaFromPlaylist(playlistId, mediaId)
    }
    
    suspend fun clearPlaylist(playlistId: Long) = withContext(Dispatchers.IO) {
        playlistMediaCrossRefDao.clearPlaylist(playlistId)
    }
    
    fun getPlaylistMediaCrossRefs(playlistId: Long): Flow<List<PlaylistMediaCrossRef>> =
        playlistMediaCrossRefDao.getPlaylistMediaCrossRefs(playlistId)
} 