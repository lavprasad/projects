package com.example.mediaplayerapp.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.net.Uri
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.support.v4.media.session.MediaSessionCompat
import android.support.v4.media.session.PlaybackStateCompat
import androidx.core.app.NotificationCompat
import com.example.mediaplayerapp.MainActivity
import com.example.mediaplayerapp.R
import com.example.mediaplayerapp.data.entity.MediaItem
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit

class MediaPlayerService : Service() {

    companion object {
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "media_player_channel"
    }

    // MediaPlayer
    private var mediaPlayer: MediaPlayer? = null
    private val binder = MediaPlayerBinder()
    
    // Playback state
    private var currentMediaItem: MediaItem? = null
    private var currentPosition = 0
    private var isPlaying = false
    
    // For tracking playback position
    private var executor: ScheduledExecutorService? = null
    
    // Callback for clients
    private var callback: MediaPlayerCallback? = null
    
    // MediaSession
    private lateinit var mediaSession: MediaSessionCompat
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize MediaSession
        mediaSession = MediaSessionCompat(this, "MediaPlayerService")
        mediaSession.setCallback(mediaSessionCallback)
        
        // Create notification channel for Android O and above
        createNotificationChannel()
    }
    
    override fun onBind(intent: Intent): IBinder {
        return binder
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, createNotification())
        return START_NOT_STICKY
    }
    
    override fun onDestroy() {
        releaseMediaPlayer()
        stopForeground(true)
        mediaSession.release()
        super.onDestroy()
    }
    
    // Initialize MediaPlayer with a MediaItem
    fun initializeMediaPlayer(mediaItem: MediaItem) {
        try {
            // Release previous MediaPlayer if it exists
            releaseMediaPlayer()
            
            // Create new MediaPlayer
            mediaPlayer = MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .build()
                )
                
                // Set data source from the MediaItem's URI
                setDataSource(applicationContext, Uri.parse(mediaItem.uri))
                
                // Prepare the MediaPlayer asynchronously
                prepareAsync()
                
                // Set listeners
                setOnPreparedListener {
                    // Start playback when prepared
                    start()
                    isPlaying = true
                    currentMediaItem = mediaItem
                    
                    // Start tracking playback position
                    startTrackingPosition()
                    
                    // Update MediaSession
                    updatePlaybackState(PlaybackStateCompat.STATE_PLAYING)
                    
                    // Update notification
                    startForeground(NOTIFICATION_ID, createNotification())
                    
                    // Notify callback
                    callback?.onMediaPlay(mediaItem)
                }
                
                setOnCompletionListener {
                    isPlaying = false
                    stopTrackingPosition()
                    updatePlaybackState(PlaybackStateCompat.STATE_STOPPED)
                    callback?.onMediaComplete()
                }
                
                setOnErrorListener { _, what, extra ->
                    isPlaying = false
                    stopTrackingPosition()
                    updatePlaybackState(PlaybackStateCompat.STATE_ERROR)
                    callback?.onError("MediaPlayer error: $what, $extra")
                    true
                }
            }
        } catch (e: IOException) {
            isPlaying = false
            callback?.onError("Error initializing MediaPlayer: ${e.message}")
        }
    }
    
    // Play or pause the MediaPlayer
    fun togglePlayPause() {
        mediaPlayer?.let {
            if (isPlaying) {
                it.pause()
                isPlaying = false
                stopTrackingPosition()
                updatePlaybackState(PlaybackStateCompat.STATE_PAUSED)
                callback?.onMediaPause()
            } else {
                it.start()
                isPlaying = true
                startTrackingPosition()
                updatePlaybackState(PlaybackStateCompat.STATE_PLAYING)
                callback?.onMediaPlay(currentMediaItem)
            }
            
            // Update notification
            startForeground(NOTIFICATION_ID, createNotification())
        }
    }
    
    // Stop playback
    fun stop() {
        mediaPlayer?.let {
            if (it.isPlaying) {
                it.stop()
                isPlaying = false
                stopTrackingPosition()
                updatePlaybackState(PlaybackStateCompat.STATE_STOPPED)
                callback?.onMediaStop()
                
                // Update notification or stop foreground
                stopForeground(true)
            }
        }
    }
    
    // Seek to a specific position
    fun seekTo(position: Int) {
        mediaPlayer?.let {
            it.seekTo(position)
            currentPosition = position
            callback?.onPositionChanged(position)
            updatePlaybackState(if (isPlaying) PlaybackStateCompat.STATE_PLAYING else PlaybackStateCompat.STATE_PAUSED)
        }
    }
    
    // Get the current playback position
    fun getCurrentPosition(): Int {
        return mediaPlayer?.currentPosition ?: 0
    }
    
    // Get the total duration
    fun getDuration(): Int {
        return mediaPlayer?.duration ?: 0
    }
    
    // Check if MediaPlayer is playing
    fun isPlaying(): Boolean {
        return isPlaying
    }
    
    // Get the current MediaItem
    fun getCurrentMediaItem(): MediaItem? {
        return currentMediaItem
    }
    
    // Set callback for playback events
    fun setCallback(callback: MediaPlayerCallback) {
        this.callback = callback
    }
    
    // Release MediaPlayer resources
    private fun releaseMediaPlayer() {
        mediaPlayer?.let {
            if (it.isPlaying) {
                it.stop()
            }
            it.release()
            mediaPlayer = null
            isPlaying = false
            stopTrackingPosition()
        }
    }
    
    // Start tracking playback position
    private fun startTrackingPosition() {
        stopTrackingPosition()
        executor = Executors.newSingleThreadScheduledExecutor()
        executor?.scheduleAtFixedRate({
            mediaPlayer?.let {
                currentPosition = it.currentPosition
                callback?.onPositionChanged(currentPosition)
            }
        }, 0, 1000, TimeUnit.MILLISECONDS)
    }
    
    // Stop tracking playback position
    private fun stopTrackingPosition() {
        executor?.shutdown()
        executor = null
    }
    
    // Create notification channel for Android O and above
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Media Player",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Media player controls"
                setShowBadge(false)
            }
            
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }
    
    // Create notification for foreground service
    private fun createNotification(): Notification {
        val mediaItem = currentMediaItem ?: return createEmptyNotification()
        
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent, PendingIntent.FLAG_IMMUTABLE
        )
        
        val playPauseIcon = if (isPlaying) android.R.drawable.ic_media_pause else android.R.drawable.ic_media_play
        
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(mediaItem.title)
            .setContentText(mediaItem.artist)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .addAction(playPauseIcon, "Play/Pause", createPlayPausePendingIntent())
            .setStyle(androidx.media.app.NotificationCompat.MediaStyle()
                .setMediaSession(mediaSession.sessionToken)
                .setShowActionsInCompactView(0))
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
    
    // Create an empty notification when no media is playing
    private fun createEmptyNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Media Player")
            .setContentText("No media playing")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
    
    // Create PendingIntent for play/pause action
    private fun createPlayPausePendingIntent(): PendingIntent {
        val intent = Intent(this, MediaPlayerService::class.java).apply {
            action = "PLAY_PAUSE"
        }
        
        return PendingIntent.getService(
            this, 0, intent, PendingIntent.FLAG_IMMUTABLE
        )
    }
    
    // Update MediaSession playback state
    private fun updatePlaybackState(state: Int) {
        val position = mediaPlayer?.currentPosition?.toLong() ?: 0L
        
        val stateBuilder = PlaybackStateCompat.Builder()
            .setActions(PlaybackStateCompat.ACTION_PLAY 
                    or PlaybackStateCompat.ACTION_PAUSE
                    or PlaybackStateCompat.ACTION_PLAY_PAUSE
                    or PlaybackStateCompat.ACTION_STOP
                    or PlaybackStateCompat.ACTION_SEEK_TO)
            .setState(state, position, 1.0f)
        
        mediaSession.setPlaybackState(stateBuilder.build())
    }
    
    // MediaSession callback
    private val mediaSessionCallback = object : MediaSessionCompat.Callback() {
        override fun onPlay() {
            if (!isPlaying) {
                togglePlayPause()
            }
        }
        
        override fun onPause() {
            if (isPlaying) {
                togglePlayPause()
            }
        }
        
        override fun onStop() {
            stop()
        }
        
        override fun onSeekTo(pos: Long) {
            seekTo(pos.toInt())
        }
    }
    
    // Binder for client communication
    inner class MediaPlayerBinder : Binder() {
        fun getService(): MediaPlayerService = this@MediaPlayerService
    }
    
    // Callback interface for clients
    interface MediaPlayerCallback {
        fun onMediaPlay(mediaItem: MediaItem?)
        fun onMediaPause()
        fun onMediaStop()
        fun onMediaComplete()
        fun onPositionChanged(position: Int)
        fun onError(errorMessage: String)
    }
} 