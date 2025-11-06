package com.example.myapplication

import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.ByteArrayOutputStream

class CameraCaptureActivity : AppCompatActivity() {


    private var startTime = 0L
    private var captured = false

    private lateinit var previewView: PreviewView
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        previewView = PreviewView(this)
        setContentView(previewView)

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()

            val preview = Preview.Builder().build()
            preview.setSurfaceProvider(previewView.surfaceProvider)

            val imageCapture = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageCapture.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
                captureFace(imageProxy)
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                imageCapture
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureFace(imageProxy: ImageProxy) {
        val media = imageProxy.image ?: return imageProxy.close()

        val input = InputImage.fromMediaImage(media, imageProxy.imageInfo.rotationDegrees)

        // Start timer the first time we see a face
        if (startTime == 0L) {
            startTime = System.currentTimeMillis()
        }

        detector.process(input)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {

                    val elapsed = System.currentTimeMillis() - startTime

                    // ✅ Wait 10 seconds before capturing
                    if (elapsed < 5000 || captured) {
                        imageProxy.close()
                        return@addOnSuccessListener
                    }

                    captured = true  // ✅ avoid double capture

                    val faceBox = faces[0].boundingBox
                    val bmp = toBitmap(imageProxy)

                    val cropped = cropFace(bmp, faceBox)

                    // return bitmap
                    val stream = ByteArrayOutputStream()
                    cropped.compress(Bitmap.CompressFormat.JPEG, 100, stream)
                    val bytes = stream.toByteArray()

                    val intent = intent
                    intent.putExtra("face_bytes", bytes)
                    setResult(RESULT_OK, intent)
                    finish()
                }
            }
            .addOnFailureListener {
                Toast.makeText(this, "Face detect error", Toast.LENGTH_SHORT).show()
            }
            .addOnCompleteListener { imageProxy.close() }
    }

    private fun toBitmap(imageProxy: ImageProxy): Bitmap {
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

        val image = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        image.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun cropFace(bmp: Bitmap, rect: Rect): Bitmap {
        val x = rect.left.coerceIn(0, bmp.width)
        val y = rect.top.coerceIn(0, bmp.height)
        val w = rect.width().coerceAtMost(bmp.width - x)
        val h = rect.height().coerceAtMost(bmp.height - y)
        return Bitmap.createBitmap(bmp, x, y, w, h)
    }
}
