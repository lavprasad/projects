package com.example.mediaplayerapp

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupWithNavController
import com.example.mediaplayerapp.data.entity.MediaItem
import com.example.mediaplayerapp.databinding.ActivityMainBinding
import com.example.mediaplayerapp.service.MediaPlayerService
import com.google.android.material.bottomnavigation.BottomNavigationView

class MainActivity : AppCompatActivity(), MediaPlayerService.MediaPlayerCallback {

    private lateinit var binding: ActivityMainBinding
    
    // Views for now playing bar
    private lateinit var nowPlayingContainer: ConstraintLayout
    private lateinit var nowPlayingTitle: TextView
    private lateinit var nowPlayingArtist: TextView
    private lateinit var playPauseButton: ImageButton
    
    // MediaPlayerService
    private var mediaPlayerService: MediaPlayerService? = null
    private var isServiceBound = false
    
    // Service connection
    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as MediaPlayerService.MediaPlayerBinder
            mediaPlayerService = binder.getService()
            mediaPlayerService?.setCallback(this@MainActivity)
            isServiceBound = true
            
            // Update UI if there's already a media playing
            updateNowPlayingBar()
        }
        
        override fun onServiceDisconnected(name: ComponentName?) {
            mediaPlayerService = null
            isServiceBound = false
        }
    }
    
    // Permission request launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions.entries.all { it.value }
        if (granted) {
            // Permissions granted, initialize app
            initializeApp()
        } else {
            // Permissions denied, show explanation
            showPermissionDeniedMessage()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Initialize views
        nowPlayingContainer = binding.nowPlayingContainer
        nowPlayingTitle = binding.nowPlayingTitle
        nowPlayingArtist = binding.nowPlayingArtist
        playPauseButton = binding.playPauseButton
        
        // Setup navigation
        val navHostFragment = supportFragmentManager.findFragmentById(R.id.nav_host_fragment) as NavHostFragment
        val navController = navHostFragment.navController
        val bottomNav = findViewById<BottomNavigationView>(R.id.bottom_navigation)
        bottomNav.setupWithNavController(navController)
        
        // Setup click listeners
        playPauseButton.setOnClickListener {
            mediaPlayerService?.togglePlayPause()
        }
        
        nowPlayingContainer.setOnClickListener {
            // TODO: Open now playing screen
        }
        
        // Check permissions
        checkPermissions()
    }
    
    override fun onStart() {
        super.onStart()
        bindMediaPlayerService()
    }
    
    override fun onStop() {
        super.onStop()
        
        if (isServiceBound) {
            unbindService(serviceConnection)
            isServiceBound = false
        }
    }
    
    // Check if we have the necessary permissions
    private fun checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // For Android 13+ (API 33+)
            val permission = Manifest.permission.READ_MEDIA_AUDIO
            
            if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
                initializeApp()
            } else {
                requestPermissionLauncher.launch(arrayOf(permission))
            }
        } else {
            // For older Android versions
            val permission = Manifest.permission.READ_EXTERNAL_STORAGE
            
            if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
                initializeApp()
            } else {
                requestPermissionLauncher.launch(arrayOf(permission))
            }
        }
    }
    
    // Initialize app after permissions are granted
    private fun initializeApp() {
        // Initialization code after permissions are granted
    }
    
    // Show message when permissions are denied
    private fun showPermissionDeniedMessage() {
        // TODO: Show a dialog explaining why we need the permissions
    }
    
    // Bind to MediaPlayerService
    private fun bindMediaPlayerService() {
        Intent(this, MediaPlayerService::class.java).also { intent ->
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        }
    }
    
    // Update the now playing bar
    private fun updateNowPlayingBar() {
        val currentMediaItem = mediaPlayerService?.getCurrentMediaItem()
        val isPlaying = mediaPlayerService?.isPlaying() ?: false
        
        if (currentMediaItem != null) {
            nowPlayingContainer.visibility = View.VISIBLE
            nowPlayingTitle.text = currentMediaItem.title
            nowPlayingArtist.text = currentMediaItem.artist
            
            // Update play/pause icon
            playPauseButton.setImageResource(
                if (isPlaying) android.R.drawable.ic_media_pause 
                else android.R.drawable.ic_media_play
            )
        } else {
            nowPlayingContainer.visibility = View.GONE
        }
    }
    
    // Play a media item
    fun playMedia(mediaItem: MediaItem) {
        // Start service if not running
        val serviceIntent = Intent(this, MediaPlayerService::class.java)
        startService(serviceIntent)
        
        // Ensure service is bound
        if (!isServiceBound) {
            bindMediaPlayerService()
        }
        
        // Initialize and play media
        mediaPlayerService?.initializeMediaPlayer(mediaItem)
    }
    
    // MediaPlayerService.MediaPlayerCallback implementation
    override fun onMediaPlay(mediaItem: MediaItem?) {
        updateNowPlayingBar()
    }
    
    override fun onMediaPause() {
        val isPlaying = mediaPlayerService?.isPlaying() ?: false
        playPauseButton.setImageResource(
            if (isPlaying) android.R.drawable.ic_media_pause 
            else android.R.drawable.ic_media_play
        )
    }
    
    override fun onMediaStop() {
        nowPlayingContainer.visibility = View.GONE
    }
    
    override fun onMediaComplete() {
        // Auto-play next track or stop
        playPauseButton.setImageResource(android.R.drawable.ic_media_play)
    }
    
    override fun onPositionChanged(position: Int) {
        // Update seekbar if implemented
    }
    
    override fun onError(errorMessage: String) {
        // Show error message
    }
} 