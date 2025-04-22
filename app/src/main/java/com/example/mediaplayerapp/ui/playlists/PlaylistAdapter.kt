package com.example.mediaplayerapp.ui.playlists

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.mediaplayerapp.R
import com.example.mediaplayerapp.data.entity.Playlist

class PlaylistAdapter(private val listener: PlaylistClickListener) :
    ListAdapter<Playlist, PlaylistAdapter.PlaylistViewHolder>(PlaylistDiffCallback()) {
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PlaylistViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_playlist, parent, false)
        return PlaylistViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: PlaylistViewHolder, position: Int) {
        val playlist = getItem(position)
        holder.bind(playlist, listener)
    }
    
    class PlaylistViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val playlistIcon: ImageView = itemView.findViewById(R.id.playlist_icon)
        private val playlistName: TextView = itemView.findViewById(R.id.playlist_name)
        private val playlistCount: TextView = itemView.findViewById(R.id.playlist_count)
        private val playlistMenu: ImageButton = itemView.findViewById(R.id.playlist_menu)
        
        fun bind(playlist: Playlist, listener: PlaylistClickListener) {
            playlistName.text = playlist.name
            playlistCount.text = "0 tracks" // This will be updated when we implement playlist details
            
            itemView.setOnClickListener {
                listener.onPlaylistClick(playlist)
            }
            
            playlistMenu.setOnClickListener {
                listener.onPlaylistOptionClick(it, playlist)
            }
        }
    }
    
    class PlaylistDiffCallback : DiffUtil.ItemCallback<Playlist>() {
        override fun areItemsTheSame(oldItem: Playlist, newItem: Playlist): Boolean {
            return oldItem.playlistId == newItem.playlistId
        }
        
        override fun areContentsTheSame(oldItem: Playlist, newItem: Playlist): Boolean {
            return oldItem == newItem
        }
    }
    
    interface PlaylistClickListener {
        fun onPlaylistClick(playlist: Playlist)
        fun onPlaylistOptionClick(view: View, playlist: Playlist)
    }
} 