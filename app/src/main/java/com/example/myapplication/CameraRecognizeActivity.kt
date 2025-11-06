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
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

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

    // Config
    private val MIRROR_FRONT = true
    private val CROP_SCALE = 1.3f
    private val DIST_THRESHOLD = 0.80f // 128-d Euclidean

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera_recognize)

        viewFinder = findViewById(R.id.viewFinder)
        faceGuide = findViewById(R.id.faceGuide)
        tvStatus = findViewById(R.id.tvStatus)
        tvLightWarning = findViewById(R.id.tvLightWarning)
        progressVerifying = findViewById(R.id.progressVerifying)
        btnClose = findViewById(R.id.btnClose)

        faceNet = FaceNetHelper(this) // expects 128-d model
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
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also { ia ->
                    ia.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val t = System.currentTimeMillis()
        if (t - lastProcessTime < 350) { // ~3 FPS analysis
            imageProxy.close()
            return
        }
        lastProcessTime = t

        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        // Build one upright + mirrored bitmap; detect on this to keep coords aligned
        val rotated = imageProxyToBitmapUpright(imageProxy)
        val prepared = if (MIRROR_FRONT) mirrorBitmap(rotated) else rotated

        // Lighting (use Y plane mean with correct divisor)
        val yBuffer: ByteBuffer = imageProxy.planes[0].buffer.duplicate()
        var sum = 0L
        val count = yBuffer.remaining()
        while (yBuffer.hasRemaining()) sum += (yBuffer.get().toInt() and 0xFF)
        val brightness = if (count > 0) sum / count else 0L

        runOnUiThread {
            tvLightWarning.visibility = if (brightness < 40) View.VISIBLE else View.GONE
        }

        val image = InputImage.fromBitmap(prepared, 0)

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    faceGuide.background.setTint(Color.GREEN)
                    val now = System.currentTimeMillis()
                    if (faceStableStart == 0L) faceStableStart = now

                    // face stable for 1s and not already verifying
                    if (now - faceStableStart > 1000 && !isVerifying) {
                        isVerifying = true
                        runOnUiThread {
                            progressVerifying.visibility = View.VISIBLE
                            tvStatus.text = "Verifying..."
                        }

                        // Use FIRST face
                        val face = faces[0]
                        val cropped = cropWithScale(prepared, face.boundingBox, CROP_SCALE)

                        // Get embedding (128-d) and recognize
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

    private fun recognizeFace(embedding: FloatArray) {
        lifecycleScope.launch {
            val faces = db.faceDao().getAllFaces()
            if (faces.isEmpty()) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "No faces enrolled"
                    progressVerifying.visibility = View.GONE
                    isVerifying = false
                }
                return@launch
            }

            var bestName = "Unknown"
            var minDist = Float.MAX_VALUE

            for (face in faces) {
                val emb = face.embedding.split(",").map { it.toFloat() }.toFloatArray()
                val dist = faceNet.calculateDistance(emb, embedding)
                if (dist < minDist) {
                    minDist = dist
                    bestName = face.name
                }
            }

            val recognized = if (minDist < DIST_THRESHOLD) bestName else "Unknown"

            withContext(Dispatchers.Main) {
                tvStatus.text = "Recognized: $recognized (distance=$minDist)"
                Log.d("CameraRecognition", "Recognized: $recognized (distance=$minDist)")
                progressVerifying.visibility = View.GONE
                isVerifying = false
                Toast.makeText(
                    this@CameraRecognizeActivity,
                    "Recognized: $recognized",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    // --- Helpers ---

    private fun imageProxyToBitmapUpright(imageProxy: ImageProxy): Bitmap {
        val nv21 = yuv420ToNv21(imageProxy)
        val yuv = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val bytes = out.toByteArray()
        val raw = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

        val rotation = imageProxy.imageInfo.rotationDegrees.toFloat()
        if (rotation == 0f) return raw
        val m = Matrix().apply { postRotate(rotation) }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
    }

    private fun yuv420ToNv21(imageProxy: ImageProxy): ByteArray {
        val y = imageProxy.planes[0].buffer
        val u = imageProxy.planes[1].buffer
        val v = imageProxy.planes[2].buffer

        val ySize = y.remaining()
        val uSize = u.remaining()
        val vSize = v.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        y.get(nv21, 0, ySize)
        v.get(nv21, ySize, vSize)
        u.get(nv21, ySize + vSize, uSize)
        return nv21
    }

    private fun mirrorBitmap(src: Bitmap): Bitmap {
        val m = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun cropWithScale(bmp: Bitmap, rect: Rect, scale: Float): Bitmap {
        val cx = rect.centerX()
        val cy = rect.centerY()
        val halfW = (rect.width() * scale / 2).toInt()
        val halfH = (rect.height() * scale / 2).toInt()

        val x = max(0, cx - halfW)
        val y = max(0, cy - halfH)
        val w = min(bmp.width - x, halfW * 2)
        val h = min(bmp.height - y, halfH * 2)

        return Bitmap.createBitmap(bmp, x, y, w, h)
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
