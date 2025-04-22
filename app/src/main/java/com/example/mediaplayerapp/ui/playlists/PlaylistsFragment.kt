package com.example.mediaplayerapp.ui.playlists

import android.app.AlertDialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.RecyclerView
import com.example.mediaplayerapp.R
import com.example.mediaplayerapp.data.entity.Playlist
import com.example.mediaplayerapp.data.entity.PlaylistWithMedia
import com.google.android.material.floatingactionbutton.FloatingActionButton

class PlaylistsFragment : Fragment(), PlaylistAdapter.PlaylistClickListener {
    
    private lateinit var viewModel: PlaylistsViewModel
    private lateinit var playlistsAdapter: PlaylistAdapter
    private lateinit var recyclerView: RecyclerView
    private lateinit var emptyView: TextView
    private lateinit var fabCreatePlaylist: FloatingActionButton

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val root = inflater.inflate(R.layout.fragment_playlists, container, false)
        
        recyclerView = root.findViewById(R.id.playlists_list)
        emptyView = root.findViewById(R.id.empty_view)
        fabCreatePlaylist = root.findViewById(R.id.fab_create_playlist)
        
        return root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        // Initialize ViewModel
        viewModel = ViewModelProvider(this).get(PlaylistsViewModel::class.java)
        
        // Setup adapter
        playlistsAdapter = PlaylistAdapter(this)
        recyclerView.adapter = playlistsAdapter
        
        // Observe playlists
        viewModel.playlists.observe(viewLifecycleOwner) { playlists ->
            if (playlists.isEmpty()) {
                recyclerView.visibility = View.GONE
                emptyView.visibility = View.VISIBLE
            } else {
                recyclerView.visibility = View.VISIBLE
                emptyView.visibility = View.GONE
                playlistsAdapter.submitList(playlists)
            }
        }
        
        // Setup FAB
        fabCreatePlaylist.setOnClickListener {
            showCreatePlaylistDialog()
        }
    }
    
    // Show dialog to create a new playlist
    private fun showCreatePlaylistDialog() {
        val dialogView = LayoutInflater.from(requireContext())
            .inflate(R.layout.dialog_create_playlist, null)
        
        val editPlaylistName = dialogView.findViewById<EditText>(R.id.edit_playlist_name)
        
        AlertDialog.Builder(requireContext())
            .setTitle(R.string.create_playlist)
            .setView(dialogView)
            .setPositiveButton(R.string.save) { _, _ ->
                val playlistName = editPlaylistName.text.toString().trim()
                if (playlistName.isNotEmpty()) {
                    viewModel.createPlaylist(playlistName)
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
    
    // Handle playlist click
    override fun onPlaylistClick(playlist: Playlist) {
        // Navigate to playlist detail fragment
        // TODO: Implement navigation to playlist detail
    }
    
    // Handle playlist option menu click
    override fun onPlaylistOptionClick(view: View, playlist: Playlist) {
        // Show popup menu with options
        // TODO: Implement popup menu with delete option
    }
} 