package com.example.mediaplayerapp.ui.media

import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mediaplayerapp.data.entity.MediaItem
import com.example.mediaplayerapp.data.entity.Playlist
import com.example.mediaplayerapp.repository.MediaRepository
import com.example.mediaplayerapp.repository.PlaylistRepository
import kotlinx.coroutines.launch

class MediaViewModel : ViewModel() {
    
    private val _mediaItems = MutableLiveData<List<MediaItem>>()
    val mediaItems: LiveData<List<MediaItem>> = _mediaItems
    
    private val mediaRepository = lazy { MediaRepository(Context) }
    private val playlistRepository = PlaylistRepository()
    
    // Load media items from device storage
    fun loadMediaItems(context: Context) {
        viewModelScope.launch {
            val repository = MediaRepository(context)
            val items = repository.getAllMediaItems()
            _mediaItems.postValue(items)
        }
    }
    
    // Get all playlists
    fun getPlaylists(): LiveData<List<Playlist>> {
        return playlistRepository.getAllPlaylists()
    }
    
    // Add media to playlist
    fun addMediaToPlaylist(playlistId: Long, mediaItem: MediaItem) {
        viewModelScope.launch {
            playlistRepository.addMediaToPlaylist(playlistId, mediaItem)
        }
    }
} 