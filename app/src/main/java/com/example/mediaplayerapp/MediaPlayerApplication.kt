package com.example.mediaplayerapp

import android.app.Application
import androidx.room.Room
import com.example.mediaplayerapp.data.AppDatabase

class MediaPlayerApplication : Application() {
    
    companion object {
        lateinit var database: AppDatabase
            private set
    }
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize the database
        database = Room.databaseBuilder(
            applicationContext,
            AppDatabase::class.java,
            "mediaplayer-database"
        ).build()
    }
} 