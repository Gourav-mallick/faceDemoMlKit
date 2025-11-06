package com.example.myapplication.database

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase

@Entity(tableName = "faces")
data class FaceEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val name: String,
    val embedding: String // store as JSON string
)



@Dao
interface FaceDao {
    @Insert
    suspend fun insert(face: FaceEntity)

    @Query("SELECT * FROM faces")
    suspend fun getAllFaces(): List<FaceEntity>

    @Query("SELECT * FROM faces WHERE LOWER(name) = LOWER(:name) LIMIT 1")
    suspend fun getFaceByName(name: String): FaceEntity?

}



@Database(entities = [FaceEntity::class], version = 1)
abstract class FaceDatabase : RoomDatabase() {
    abstract fun faceDao(): FaceDao

    companion object {
        private var INSTANCE: FaceDatabase? = null

        fun getDatabase(context: Context): FaceDatabase {
            if (INSTANCE == null) {
                INSTANCE = Room.databaseBuilder(
                    context.applicationContext,
                    FaceDatabase::class.java,
                    "face_db"
                ).build()
            }
            return INSTANCE!!
        }
    }
}
