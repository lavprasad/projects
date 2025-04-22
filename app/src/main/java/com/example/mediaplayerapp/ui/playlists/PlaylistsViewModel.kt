package com.example.mediaplayerapp.ui.playlists

import androidx.lifecycle.LiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mediaplayerapp.data.entity.Playlist
import com.example.mediaplayerapp.data.entity.PlaylistWithMedia
import com.example.mediaplayerapp.repository.PlaylistRepository
import kotlinx.coroutines.launch

class PlaylistsViewModel : ViewModel() {
    
    private val playlistRepository = PlaylistRepository()
    
    // LiveData for playlists
    val playlists: LiveData<List<Playlist>> = playlistRepository.getAllPlaylists()
    
    // Create a new playlist
    fun createPlaylist(name: String) {
        viewModelScope.launch {
            playlistRepository.createPlaylist(name)
        }
    }
    
    // Delete a playlist
    fun deletePlaylist(playlist: Playlist) {
        viewModelScope.launch {
            playlistRepository.deletePlaylist(playlist)
        }
    }
    
    // Get playlist with media
    fun getPlaylistWithMedia(playlistId: Long): LiveData<PlaylistWithMedia> {
        return playlistRepository.getPlaylistWithMedia(playlistId)
    }
} 