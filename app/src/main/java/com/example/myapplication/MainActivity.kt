package com.example.myapplication

import android.graphics.Bitmap
import android.graphics.Rect
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.myapplication.database.FaceDatabase
import com.example.myapplication.database.FaceEntity
import com.example.myapplication.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var faceNet: FaceNetHelper
    private lateinit var db: FaceDatabase
    private val faceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
    )

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { processImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        faceNet = FaceNetHelper(this)
        db = FaceDatabase.getDatabase(this)

        binding.btnAddFace.setOnClickListener { pickImage.launch("image/*") }
        binding.btnRecognize.setOnClickListener { pickImage.launch("image/*") }
    }

    private fun processImage(uri: Uri) {
        val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
        val input = InputImage.fromBitmap(bitmap, 0)
        faceDetector.process(input)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    val face = faces[0]
                    val faceBitmap = cropFace(bitmap, face.boundingBox)
                    val embedding = faceNet.getFaceEmbedding(faceBitmap)
                    if (binding.switchMode.isChecked) {
                        saveFace(binding.etName.text.toString(), embedding)
                    } else {
                        recognizeFace(embedding)
                    }
                } else {
                    Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show()
                }
            }
            .addOnFailureListener {
                Toast.makeText(this, "Error detecting face: ${it.message}", Toast.LENGTH_SHORT).show()
            }
    }

    private fun cropFace(bitmap: Bitmap, rect: Rect): Bitmap {
        val x = rect.left.coerceAtLeast(0)
        val y = rect.top.coerceAtLeast(0)
        val width = rect.width().coerceAtMost(bitmap.width - x)
        val height = rect.height().coerceAtMost(bitmap.height - y)
        return Bitmap.createBitmap(bitmap, x, y, width, height)
    }

    private fun saveFace(name: String, embedding: FloatArray) {
        lifecycleScope.launch {
            db.faceDao().insert(FaceEntity(name = name, embedding = embedding.joinToString(",")))
            withContext(Dispatchers.Main) {
                Toast.makeText(this@MainActivity, "Face saved: $name", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun recognizeFace(embedding: FloatArray) {
        lifecycleScope.launch {
            val faces = db.faceDao().getAllFaces()
            var bestName = "Unknown"
            var minDist = Float.MAX_VALUE

            for (face in faces) {
                val emb = face.embedding.split(",").map { it.toFloat() }.toFloatArray()
                val dist = faceNet.calculateDistance(emb, embedding)
                Log.d("FaceRecognition", "Comparing with ${face.name}, distance = $dist")

                if (dist < minDist) {
                    minDist = dist
                    bestName = face.name
                }
            }

            // Apply threshold AFTER loop
            val threshold = 0.8f
            if (minDist > threshold) {
                bestName = "Unknown"
            }

            withContext(Dispatchers.Main) {
                Toast.makeText(
                    this@MainActivity,
                    "Recognized: $bestName (distance=$minDist)",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
}
