package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.myapplication.database.FaceDatabase
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraRecognizeActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var faceNet: FaceNetHelper
    private lateinit var db: FaceDatabase

    private lateinit var viewFinder: PreviewView
    private lateinit var faceGuide: View
    private lateinit var tvStatus: TextView
    private lateinit var tvLightWarning: TextView
    private lateinit var btnClose: Button
    private lateinit var progressVerifying: ProgressBar

    private var faceStableStart = 0L
    private var isVerifying = false
    private var lastProcessTime = 0L

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera_recognize)

        // Initialize UI
        viewFinder = findViewById(R.id.viewFinder)
        faceGuide = findViewById(R.id.faceGuide)
        tvStatus = findViewById(R.id.tvStatus)
        tvLightWarning = findViewById(R.id.tvLightWarning)
        progressVerifying = findViewById(R.id.progressVerifying)
        btnClose = findViewById(R.id.btnClose)

        faceNet = FaceNetHelper(this)
        db = FaceDatabase.getDatabase(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        btnClose.setOnClickListener { finish() }

        if (allPermissionsGranted()) startCamera()
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, 10)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processFrame(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
            preview.setSurfaceProvider(viewFinder.surfaceProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastProcessTime < 500) { // 2 frames per sec for smoothness
            imageProxy.close()
            return
        }
        lastProcessTime = currentTime

        val mediaImage = imageProxy.image ?: return imageProxy.close()
        val image = InputImage.fromMediaImage(mediaImage, 0)

        // 💡 Lighting detection
        val brightness = imageProxy.planes[0].buffer.asReadOnlyBuffer().let { buffer ->
            var sum = 0L
            while (buffer.hasRemaining()) sum += (buffer.get().toInt() and 0xFF)
            sum / imageProxy.planes[0].buffer.limit()
        }

        runOnUiThread {
            if (brightness < 40) tvLightWarning.visibility = View.VISIBLE
            else tvLightWarning.visibility = View.GONE
        }

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    faceGuide.background.setTint(Color.GREEN)
                    val now = System.currentTimeMillis()

                    if (faceStableStart == 0L) faceStableStart = now

                    // Face stable for 1 second
                    if (now - faceStableStart > 1000 && !isVerifying) {
                        isVerifying = true
                        runOnUiThread {
                            progressVerifying.visibility = View.VISIBLE
                            tvStatus.text = "Verifying..."
                        }

                        val bitmap = imageProxyToBitmap(imageProxy)
                        val mirrored = Bitmap.createBitmap(
                            bitmap, 0, 0, bitmap.width, bitmap.height,
                            Matrix().apply { preScale(-1f, 1f) }, true
                        )
                        val face = faces[0]
                        val cropped = cropFace(mirrored, face.boundingBox)
                        val embedding = faceNet.getFaceEmbedding(cropped)

                        recognizeFace(embedding)
                        faceStableStart = 0L
                    }
                } else {
                    faceGuide.background.setTint(Color.RED)
                    faceStableStart = 0L
                }
            }
            .addOnFailureListener { e ->
                Log.e("CameraRecognition", "Face detection error: ${e.message}")
            }
            .addOnCompleteListener { imageProxy.close() }
    }

    private fun cropFace(bitmap: Bitmap, rect: Rect): Bitmap {
        val scale = 1.3f
        val cx = rect.centerX()
        val cy = rect.centerY()
        val halfWidth = (rect.width() * scale / 2).toInt()
        val halfHeight = (rect.height() * scale / 2).toInt()

        val x = (cx - halfWidth).coerceAtLeast(0)
        val y = (cy - halfHeight).coerceAtLeast(0)
        val width = (halfWidth * 2).coerceAtMost(bitmap.width - x)
        val height = (halfHeight * 2).coerceAtMost(bitmap.height - y)

        return Bitmap.createBitmap(bitmap, x, y, width, height)
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun recognizeFace(embedding: FloatArray) {
        lifecycleScope.launch {
            val faces = db.faceDao().getAllFaces()
            if (faces.isEmpty()) {
                Log.d("CameraRecognition", "Database empty.")
                return@launch
            }

            var bestName = "Unknown"
            var minDist = Float.MAX_VALUE

            for (face in faces) {
                val emb = face.embedding.split(",").map { it.toFloat() }.toFloatArray()
                val dist = faceNet.calculateDistance(emb, embedding)
                Log.d("CameraRecognition", "Comparing with ${face.name}, distance = $dist")
                if (dist < minDist) {
                    minDist = dist
                    bestName = face.name
                }
            }

            val threshold = 1.15f
            val recognizedName = if (minDist < threshold) bestName else "Unknown"

            withContext(Dispatchers.Main) {
                tvStatus.text = "Recognized: $recognizedName (distance=$minDist)"
                progressVerifying.visibility = View.GONE
                isVerifying = false
                Toast.makeText(this@CameraRecognizeActivity, "Recognized: $recognizedName", Toast.LENGTH_SHORT).show()
                Log.d("CameraRecognition", "Result => Name: $recognizedName, Distance: $minDist")
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }




    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }



    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
