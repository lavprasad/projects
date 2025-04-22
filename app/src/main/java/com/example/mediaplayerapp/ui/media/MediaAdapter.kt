package com.example.mediaplayerapp.ui.media

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
import com.example.mediaplayerapp.data.entity.MediaItem
import java.util.concurrent.TimeUnit

class MediaAdapter(private val listener: MediaItemListener) : 
    ListAdapter<MediaItem, MediaAdapter.MediaViewHolder>(MediaDiffCallback()) {
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MediaViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_media, parent, false)
        return MediaViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: MediaViewHolder, position: Int) {
        val mediaItem = getItem(position)
        holder.bind(mediaItem, listener)
    }
    
    class MediaViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val mediaIcon: ImageView = itemView.findViewById(R.id.media_icon)
        private val mediaTitle: TextView = itemView.findViewById(R.id.media_title)
        private val mediaArtist: TextView = itemView.findViewById(R.id.media_artist)
        private val mediaDuration: TextView = itemView.findViewById(R.id.media_duration)
        private val mediaMenu: ImageButton = itemView.findViewById(R.id.media_menu)
        
        fun bind(mediaItem: MediaItem, listener: MediaItemListener) {
            // Set data
            mediaTitle.text = mediaItem.title
            mediaArtist.text = mediaItem.artist ?: "Unknown Artist"
            mediaDuration.text = formatDuration(mediaItem.duration)
            
            // Set click listeners
            itemView.setOnClickListener {
                listener.onMediaItemClick(mediaItem)
            }
            
            mediaMenu.setOnClickListener {
                listener.onMediaItemMenuClick(it, mediaItem)
            }
        }
        
        private fun formatDuration(durationMs: Long): String {
            val minutes = TimeUnit.MILLISECONDS.toMinutes(durationMs)
            val seconds = TimeUnit.MILLISECONDS.toSeconds(durationMs) -
                    TimeUnit.MINUTES.toSeconds(minutes)
            return String.format("%d:%02d", minutes, seconds)
        }
    }
    
    class MediaDiffCallback : DiffUtil.ItemCallback<MediaItem>() {
        override fun areItemsTheSame(oldItem: MediaItem, newItem: MediaItem): Boolean {
            return oldItem.mediaId == newItem.mediaId
        }
        
        override fun areContentsTheSame(oldItem: MediaItem, newItem: MediaItem): Boolean {
            return oldItem == newItem
        }
    }
    
    interface MediaItemListener {
        fun onMediaItemClick(mediaItem: MediaItem)
        fun onMediaItemMenuClick(view: View, mediaItem: MediaItem)
    }
} 