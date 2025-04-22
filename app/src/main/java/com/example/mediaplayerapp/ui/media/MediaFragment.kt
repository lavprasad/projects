package com.example.mediaplayerapp.ui.media

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.PopupMenu
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.RecyclerView
import com.example.mediaplayerapp.MainActivity
import com.example.mediaplayerapp.R
import com.example.mediaplayerapp.data.entity.MediaItem
import com.example.mediaplayerapp.data.entity.Playlist
import com.example.mediaplayerapp.repository.PlaylistRepository

class MediaFragment : Fragment(), MediaAdapter.MediaItemListener {

    private lateinit var viewModel: MediaViewModel
    private lateinit var mediaAdapter: MediaAdapter
    private lateinit var mediaList: RecyclerView
    private lateinit var emptyView: TextView
    
    private val playlistRepository = PlaylistRepository()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val root = inflater.inflate(R.layout.fragment_media, container, false)
        
        mediaList = root.findViewById(R.id.media_list)
        emptyView = root.findViewById(R.id.empty_view)
        
        return root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        // Setup ViewModel
        viewModel = ViewModelProvider(this).get(MediaViewModel::class.java)
        
        // Setup RecyclerView adapter
        mediaAdapter = MediaAdapter(this)
        mediaList.adapter = mediaAdapter
        
        // Observe media items
        viewModel.mediaItems.observe(viewLifecycleOwner) { mediaItems ->
            if (mediaItems.isEmpty()) {
                mediaList.visibility = View.GONE
                emptyView.visibility = View.VISIBLE
            } else {
                mediaList.visibility = View.VISIBLE
                emptyView.visibility = View.GONE
                mediaAdapter.submitList(mediaItems)
            }
        }
        
        // Load media items
        viewModel.loadMediaItems(requireContext())
    }
    
    // Handle media item click
    override fun onMediaItemClick(mediaItem: MediaItem) {
        (activity as? MainActivity)?.playMedia(mediaItem)
    }
    
    // Handle media item menu click
    override fun onMediaItemMenuClick(view: View, mediaItem: MediaItem) {
        showPopupMenu(view, mediaItem)
    }
    
    // Show popup menu for media item
    private fun showPopupMenu(view: View, mediaItem: MediaItem) {
        val popupMenu = PopupMenu(requireContext(), view)
        popupMenu.menuInflater.inflate(R.menu.media_item_menu, popupMenu.menu)
        
        // Observe playlists to add to menu
        viewModel.getPlaylists().observe(viewLifecycleOwner) { playlists ->
            if (playlists.isNotEmpty()) {
                val addToPlaylistItem = popupMenu.menu.findItem(R.id.action_add_to_playlist)
                val subMenu = addToPlaylistItem.subMenu
                
                // Clear previous items
                subMenu.clear()
                
                // Add playlists as submenu items
                for (playlist in playlists) {
                    subMenu.add(playlist.name).setOnMenuItemClickListener {
                        addMediaToPlaylist(mediaItem, playlist)
                        true
                    }
                }
            }
        }
        
        popupMenu.setOnMenuItemClickListener { menuItem ->
            when (menuItem.itemId) {
                R.id.action_play -> {
                    onMediaItemClick(mediaItem)
                    true
                }
                else -> false
            }
        }
        
        popupMenu.show()
    }
    
    // Add media to a playlist
    private fun addMediaToPlaylist(mediaItem: MediaItem, playlist: Playlist) {
        viewModel.addMediaToPlaylist(playlist.playlistId, mediaItem)
    }
}